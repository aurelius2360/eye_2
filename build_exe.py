import os
import sys
import urllib.request

def download_if_missing(filename, url):
    """Downloads model files if they are not already in the directory for bundling."""
    if not os.path.exists(filename):
        print(f"[Build] Model file '{filename}' not found. Downloading for bundle...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"[Build] Download complete: {filename}")
        except Exception as e:
            print(f"[Build] Error downloading {filename}: {e}")
            sys.exit(1)

# Ensure models are locally cached so they can be bundled inside the single EXE file
download_if_missing(
    "face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
download_if_missing(
    "efficientdet_lite0.tflite",
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
)

# Try running PyInstaller
try:
    import PyInstaller.__main__
except ImportError:
    print("\n[Build] Error: 'pyinstaller' is not installed in your Python environment.")
    print("[Build] Please run: pip install pyinstaller")
    sys.exit(1)

# Try to dynamically locate and bundle the pre-compiled FFmpeg binary
ffmpeg_add_data = []
try:
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(ffmpeg_bin):
        ffmpeg_add_data = [f'--add-data={ffmpeg_bin};.']
        print(f"[Build] Found pre-compiled FFmpeg binary: '{ffmpeg_bin}'. Bundling into executable.")
except ImportError:
    print("[Build] Warning: 'imageio-ffmpeg' is not installed. FFmpeg binary will not be bundled.")

print("\n[Build] Starting PyInstaller build process...")
pyinstaller_args = [
    'EyePupilTracker.py',
    '--onefile',
    '--console',
    '--name=EyePupilTracker',
    '--add-data=face_landmarker.task;.',
    '--add-data=efficientdet_lite0.tflite;.',
] + ffmpeg_add_data + [
    '--hidden-import=ObjectDetectorWrapper',
    '--hidden-import=PreRollPostRollRecorder',
    '--hidden-import=DeviceProctorScanner',
    '--hidden-import=bleak',
    '--hidden-import=winrt',
    '--hidden-import=mediapipe',
    '--hidden-import=cv2',
    '--hidden-import=ffmpeg',
]

PyInstaller.__main__.run(pyinstaller_args)

print("\n[Build] Build complete! You can find your standalone executable inside the 'dist' folder:")
print("[Build] -> dist/EyePupilTracker.exe")
