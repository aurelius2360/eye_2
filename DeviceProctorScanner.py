import asyncio
import os
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

try:
    from bleak import BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False


class DeviceProctorScanner:
    """
    Scans for nearby Bluetooth Low Energy (BLE) devices and local Wi-Fi / network devices.
    Establishes an initial pre-exam baseline and audits every 10 minutes to detect new unauthorized devices.
    """
    def __init__(self, ble_rssi_threshold=-90, audit_interval=600.0):
        self.ble_rssi_threshold = ble_rssi_threshold  # -90 dBm correlates to approx 10-30 meters
        self.audit_interval = audit_interval          # Default 10 minutes (600 seconds)

        # Baseline storage (known devices before exam start)
        self.baseline_ble = set()          # Set of BLE MAC addresses / identifiers
        self.baseline_wifi_macs = set()     # Set of MAC addresses on local Wi-Fi network
        self.baseline_ssids = set()         # Set of BSSIDs / SSIDs detected

        # Current scan results
        self.current_ble = set()
        self.current_wifi_macs = set()
        self.current_ssids = set()

        # Detected new unauthorized devices
        self.new_ble_devices = []
        self.new_wifi_devices = []
        self.new_ssids = []

        # State flags & timing
        self.baseline_completed = False
        self.is_scanning = False
        self.last_scan_time = 0.0
        self.last_scan_duration = 0.0
        self.alarm_triggered = False
        self.alarm_message = ""
        self.scan_count = 0
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()

    # ---------------------------------------------------------
    # 1. WI-FI & LOCAL NETWORK DISCOVERY
    # ---------------------------------------------------------
    def _get_local_ip_subnet(self):
        """Returns the local IP address and /24 subnet range."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            parts = local_ip.split(".")
            subnet_prefix = f"{parts[0]}.{parts[1]}.{parts[2]}."
            return local_ip, subnet_prefix
        except Exception:
            return "127.0.0.1", "192.168.1."

    def _ping_ip(self, ip):
        """Sends a quick socket ping / connection test to warm ARP table."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.05)
            # Try port 80 or 445 quick connect
            sock.connect_ex((ip, 80))
            sock.close()
        except Exception:
            pass

    def _warm_arp_cache(self, subnet_prefix):
        """Pings subnet hosts in parallel to update Windows ARP table quickly."""
        with ThreadPoolExecutor(max_workers=50) as executor:
            ips = [f"{subnet_prefix}{i}" for i in range(1, 255)]
            executor.map(self._ping_ip, ips)

    def scan_wifi_network(self):
        """
        Scans local Wi-Fi network using ARP cache analysis and returns list of (IP, MAC) pairs.
        Also scans nearby SSIDs/BSSIDs via netsh on Windows.
        """
        _, subnet_prefix = self._get_local_ip_subnet()
        # Warm up ARP table in parallel
        self._warm_arp_cache(subnet_prefix)

        wifi_devices = {}  # MAC -> IP mapping
        ssids = set()

        # Parse ARP table output
        try:
            output = subprocess.check_output(["arp", "-a"], text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            mac_regex = re.compile(r'([0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2})')
            ip_regex = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')

            for line in output.splitlines():
                ip_match = ip_regex.search(line)
                mac_match = mac_regex.search(line)
                if ip_match and mac_match:
                    ip_str = ip_match.group(1)
                    mac_str = mac_match.group(1).upper().replace('-', ':')
                    # Ignore multicast/broadcast MAC addresses (01:00:5E, FF:FF:FF)
                    if not mac_str.startswith("01:00:5E") and mac_str != "FF:FF:FF:FF:FF:FF":
                        wifi_devices[mac_str] = ip_str
        except Exception as e:
            print(f"[DeviceScanner] ARP scan error: {e}")

        # Scan nearby Wi-Fi SSIDs/BSSIDs on Windows via netsh
        if os.name == 'nt':
            try:
                cmd_out = subprocess.check_output(["netsh", "wlan", "show", "networks", "mode=bssid"],
                                                 text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                for line in cmd_out.splitlines():
                    line_str = line.strip()
                    if line_str.startswith("SSID") or line_str.startswith("BSSID"):
                        parts = line_str.split(":", 1)
                        if len(parts) == 2:
                            val = parts[1].strip()
                            if val and val != "1":
                                ssids.add(val)
            except Exception:
                pass

        return wifi_devices, ssids

    # ---------------------------------------------------------
    # 2. BLE DISCOVERY (USING BLEAK)
    # ---------------------------------------------------------
    def scan_ble_devices(self, timeout_sec=4.0):
        """Scans for nearby BLE devices with RSSI >= ble_rssi_threshold."""
        ble_found = {}  # MAC -> (Name, RSSI)
        if not BLEAK_AVAILABLE:
            print("[DeviceScanner] Bleak package not available for BLE scan.")
            return ble_found

        async def _async_ble_scan():
            try:
                devices = await BleakScanner.discover(timeout=timeout_sec, return_adv=True)
                for key, (device, adv_data) in devices.items():
                    rssi = adv_data.rssi if adv_data else getattr(device, 'rssi', -99)
                    if rssi is not None and rssi >= self.ble_rssi_threshold:
                        name = device.name or adv_data.local_name or "Unknown BLE Device"
                        address = device.address.upper()
                        ble_found[address] = (name, rssi)
            except Exception as e:
                print(f"[DeviceScanner] Async BLE scan error: {e}")

        try:
            # Create a clean event loop for execution in background thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_async_ble_scan())
            loop.close()
        except Exception as e:
            print(f"[DeviceScanner] BLE scan thread loop error: {e}")

        return ble_found

    # ---------------------------------------------------------
    # 3. COMBINED SCAN & AUDIT LOGIC
    # ---------------------------------------------------------
    def _execute_full_scan(self, is_baseline=False):
        """Internal synchronous scan routine combining BLE & Wi-Fi."""
        start_t = time.time()
        print(f"[DeviceScanner] Starting {'BASELINE' if is_baseline else 'AUDIT'} scan #{self.scan_count + 1}...")

        # Run scans
        wifi_devices, ssids = self.scan_wifi_network()
        ble_devices = self.scan_ble_devices(timeout_sec=4.0)

        elapsed = time.time() - start_t

        with self._lock:
            self.scan_count += 1
            self.last_scan_time = time.time()
            self.last_scan_duration = elapsed

            self.current_ble = set(ble_devices.keys())
            self.current_wifi_macs = set(wifi_devices.keys())
            self.current_ssids = ssids

            if is_baseline:
                self.baseline_ble = set(self.current_ble)
                self.baseline_wifi_macs = set(self.current_wifi_macs)
                self.baseline_ssids = set(self.current_ssids)
                self.baseline_completed = True
                self.alarm_triggered = False
                self.alarm_message = ""
                print(f"[DeviceScanner] Baseline Scan Complete! "
                      f"BLE: {len(self.baseline_ble)} | Wi-Fi LAN: {len(self.baseline_wifi_macs)} | SSIDs: {len(self.baseline_ssids)} ({elapsed:.2f}s)")
            else:
                # Compare current scan against baseline to detect NEW unauthorized devices
                new_ble_macs = self.current_ble - self.baseline_ble
                new_wifi_macs = self.current_wifi_macs - self.baseline_wifi_macs
                
                self.new_ble_devices = [(mac, ble_devices.get(mac, ("Unknown", -99))) for mac in new_ble_macs]
                self.new_wifi_devices = [(mac, wifi_devices.get(mac, "Unknown IP")) for mac in new_wifi_macs]

                if self.new_ble_devices or self.new_wifi_devices:
                    self.alarm_triggered = True
                    details = []
                    if self.new_ble_devices:
                        details.append(f"{len(self.new_ble_devices)} BLE Device(s)")
                    if self.new_wifi_devices:
                        details.append(f"{len(self.new_wifi_devices)} Wi-Fi LAN Device(s)")
                    self.alarm_message = f"ALARM: NEW NEARBY DEVICE DETECTED ({', '.join(details)})"
                    print(f"[DeviceScanner] ⚠️ VIOLATION: {self.alarm_message}")
                else:
                    self.alarm_triggered = False
                    self.alarm_message = ""
                    print(f"[DeviceScanner] Audit Scan Clean. No new unauthorized devices detected ({elapsed:.2f}s).")

            self.is_scanning = False

    def start_baseline_scan_async(self):
        """Launches pre-exam baseline scan in a non-blocking background thread."""
        with self._lock:
            if self.is_scanning:
                return
            self.is_scanning = True

        t = threading.Thread(target=self._execute_full_scan, kwargs={'is_baseline': True}, daemon=True)
        t.start()

    def start_audit_scan_async(self):
        """Launches 10-minute periodic audit scan in a non-blocking background thread."""
        with self._lock:
            if self.is_scanning:
                return
            self.is_scanning = True

        t = threading.Thread(target=self._execute_full_scan, kwargs={'is_baseline': False}, daemon=True)
        t.start()

    def check_and_trigger_periodic_audit(self, current_time):
        """Checks if 10 minutes have elapsed since last scan and triggers background audit."""
        if not self.baseline_completed:
            return
        if self.is_scanning:
            return

        if self.last_scan_time == 0.0 or (current_time - self.last_scan_time >= self.audit_interval):
            self.start_audit_scan_async()

    # ---------------------------------------------------------
    # 4. STATUS & GETTERS FOR MAIN APPLICATION
    # ---------------------------------------------------------
    def get_status(self):
        """Returns thread-safe dictionary of current scanner status."""
        with self._lock:
            return {
                'baseline_completed': self.baseline_completed,
                'is_scanning': self.is_scanning,
                'last_scan_time': self.last_scan_time,
                'scan_count': self.scan_count,
                'ble_count': len(self.current_ble),
                'wifi_count': len(self.current_wifi_macs),
                'baseline_ble_count': len(self.baseline_ble),
                'baseline_wifi_count': len(self.baseline_wifi_macs),
                'alarm_triggered': self.alarm_triggered,
                'alarm_message': self.alarm_message,
                'new_ble_count': len(self.new_ble_devices),
                'new_wifi_count': len(self.new_wifi_devices)
            }

    def clear_alarm(self):
        """Resets alarm state."""
        with self._lock:
            self.alarm_triggered = False
            self.alarm_message = ""
