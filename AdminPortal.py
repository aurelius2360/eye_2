import os
import sys
import json
import socket
import shutil
import urllib.parse
import webbrowser
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from model_utils import get_resource_path
from PreRollPostRollRecorder import load_config, save_config, ACTIVE_SESSION_FILE

PORT = 8000


def scan_all_sessions():
    """Scans the 'sessions/' directory to aggregate candidate session records."""
    sessions = []
    if os.path.exists("sessions"):
        for entry in os.listdir("sessions"):
            meta_file = os.path.join("sessions", entry, "session_meta.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        sessions.append(json.load(f))
                except Exception as e:
                    print(f"Error reading {meta_file}: {e}")
    sessions.sort(key=lambda s: s.get("start_time", ""), reverse=True)
    return sessions


def get_active_session_info():
    """Reads active session metadata if an exam is currently being proctored."""
    if os.path.exists(ACTIVE_SESSION_FILE):
        try:
            with open(ACTIVE_SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("is_live", False):
                    return data
        except Exception:
            pass
    return {"is_live": False}


class ProctorAdminHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        pass

    def handle(self):
        """Handle incoming connection and suppress client abort socket errors."""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.error):
            pass

    def send_json_response(self, data, status=200):
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.error):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # REST API Endpoints
        if path == "/api/config":
            self.send_json_response(load_config())
            return

        elif path == "/api/sessions":
            sessions = scan_all_sessions()
            self.send_json_response({"sessions": sessions, "count": len(sessions)})
            return

        elif path.startswith("/api/session/"):
            sess_id = path.replace("/api/session/", "").strip()
            sessions = scan_all_sessions()
            matched = next((s for s in sessions if s.get("session_id") == sess_id), None)
            if matched:
                self.send_json_response(matched)
            else:
                self.send_json_response({"error": "Session not found"}, 404)
            return

        # Live Proctoring Stream & Telemetry Endpoints
        elif path == "/api/live":
            self.send_json_response(get_active_session_info())
            return

        elif path == "/api/live/feed":
            act = get_active_session_info()
            if act.get("is_live") and act.get("session_dir"):
                feed_path = os.path.join(act["session_dir"], "live_feed.jpg")
                self.serve_media_file(feed_path)
            else:
                self.send_error(404, "No live feed active")
            return

        elif path == "/api/live/gaze":
            act = get_active_session_info()
            if act.get("is_live") and act.get("session_dir"):
                gaze_path = os.path.join(act["session_dir"], "live_gaze.jpg")
                self.serve_media_file(gaze_path)
            else:
                self.send_error(404, "No live gaze chart active")
            return

        elif path == "/api/live/telemetry":
            act = get_active_session_info()
            if act.get("is_live") and act.get("session_dir"):
                tel_path = os.path.join(act["session_dir"], "live_telemetry.json")
                if os.path.exists(tel_path):
                    try:
                        with open(tel_path, "r", encoding="utf-8") as f:
                            self.send_json_response(json.load(f))
                        return
                    except Exception:
                        pass
            self.send_json_response({"is_live": False})
            return

        # Media Resolver Endpoint
        elif path.startswith("/api/media/"):
            rel_path = urllib.parse.unquote(path.replace("/api/media/", ""))
            self.serve_media_file(rel_path)
            return

        # Static Dashboard SPA
        elif path in ["/", "/index.html", "/dashboard"]:
            self.serve_static_index()
            return

        elif any(path.lower().endswith(ext) for ext in [".webm", ".mp4", ".jpg", ".jpeg", ".png"]):
            clean_rel = urllib.parse.unquote(path.lstrip("/"))
            self.serve_media_file(clean_rel)
            return

        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/config":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                new_cfg = json.loads(body.decode("utf-8"))
                current = load_config()
                current.update(new_cfg)
                save_config(current)
                self.send_json_response({"status": "success", "config": current})
            except Exception as e:
                self.send_json_response({"error": str(e)}, 400)
            return

        elif path == "/api/launch-exam":
            try:
                subprocess.Popen([sys.executable, "EyePupilTracker.py"])
                self.send_json_response({"status": "launched", "message": "Proctoring candidate exam started."})
            except Exception as e:
                self.send_json_response({"error": f"Failed to start exam: {e}"}, 500)
            return

        else:
            self.send_error(404)

    def serve_static_index(self):
        """Serves static/index.html with bundle path support."""
        index_file = get_resource_path(os.path.join("static", "index.html"))
        if not os.path.exists(index_file):
            index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")

        if os.path.exists(index_file):
            try:
                with open(index_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(content)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.error):
                pass
        else:
            self.send_error(404, "Static index.html not found.")

    def serve_media_file(self, rel_path):
        """Serves image or streaming video supporting HTTP 206 Partial Content range requests."""
        target_path = rel_path if os.path.exists(rel_path) else os.path.join("sessions", rel_path)
        if not os.path.exists(target_path):
            try:
                self.send_error(404, f"Media file '{rel_path}' not found")
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.error):
                pass
            return

        file_size = os.path.getsize(target_path)
        ext = os.path.splitext(target_path)[1].lower()

        content_types = {
            ".webm": "video/webm",
            ".mp4": "video/mp4",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".json": "application/json"
        }
        content_type = content_types.get(ext, "application/octet-stream")

        try:
            # Range header handling for smooth timeline video scrubbing
            range_header = self.headers.get("Range", None)
            if range_header and range_header.startswith("bytes="):
                ranges = range_header.replace("bytes=", "").split("-")
                start = int(ranges[0])
                end = int(ranges[1]) if ranges[1] else file_size - 1
                if end >= file_size:
                    end = file_size - 1
                chunk_len = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(chunk_len))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                with open(target_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(chunk_len))
                return

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            with open(target_path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.error):
            pass


def main():
    server = ThreadingHTTPServer(('0.0.0.0', PORT), ProctorAdminHandler)
    url = f"http://localhost:{PORT}"
    print("\n=======================================================")
    print("  PROCTORVISION ADMIN PORTAL RUNNING")
    print(f"  Access URL: {url}")
    print("=======================================================\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAdmin Portal stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
