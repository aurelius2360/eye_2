import os
import sys
from model_utils import ensure_model_file, FACE_LANDMARKER_URL, OBJECT_DETECTOR_URL

# Ensure models are locally cached so they can be bundled inside the single EXE file
ensure_model_file("face_landmarker.task", FACE_LANDMARKER_URL)
ensure_model_file("efficientdet_lite0.tflite", OBJECT_DETECTOR_URL)

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
    '--add-data=static;static',
] + ffmpeg_add_data + [
    '--hidden-import=PreRollPostRollRecorder',
    '--hidden-import=model_utils',
    '--hidden-import=DeviceProctorScanner',
    '--hidden-import=mediapipe',
    '--hidden-import=cv2',
]

PyInstaller.__main__.run(pyinstaller_args)

print("\n[Build] Build complete! Standalone executable inside 'dist/EyePupilTracker.exe'")
