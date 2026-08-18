import sys
import subprocess
import os

def print_banner():
    print("""
============================================================
              PROCTORVISION AI PROCTORING SYSTEM
============================================================
1. Start Candidate Proctoring Exam (User / Student Side)
2. Start Admin Review Portal (Web Dashboard on http://localhost:8000)
3. Start Both (Launch Admin Portal + Candidate Exam)
4. Exit
============================================================
""")

def main():
    while True:
        print_banner()
        choice = input("Select an option (1-4): ").strip()

        if choice == '1':
            print("\n[Launcher] Starting Candidate Exam Client...")
            subprocess.run([sys.executable, "EyePupilTracker.py"])
        elif choice == '2':
            print("\n[Launcher] Starting Admin Portal (Press Ctrl+C to stop)...")
            subprocess.run([sys.executable, "AdminPortal.py"])
        elif choice == '3':
            print("\n[Launcher] Starting Admin Portal and Candidate Exam...")
            # Launch admin portal in background
            subprocess.Popen([sys.executable, "AdminPortal.py"])
            # Launch exam in foreground
            subprocess.run([sys.executable, "EyePupilTracker.py"])
        elif choice == '4' or choice.lower() in ['q', 'exit']:
            print("\nExiting ProctorVision. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
