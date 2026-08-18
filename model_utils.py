import os
import sys
import urllib.request

def get_resource_path(relative_path):
    """
    Get absolute path to resource, supporting PyInstaller bundles and dev environments.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def ensure_model_file(filename, url):
    """
    Ensures a required model file exists locally, downloading it if missing.
    """
    path = get_resource_path(filename)
    if not os.path.exists(path):
        print(f"[Model] Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"[Model] Downloaded {filename} successfully.")
        except Exception as e:
            print(f"[Model] Error downloading {filename}: {e}")
            raise
    return path

# Default Google Cloud Storage model URLs & pre-resolved static paths
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
OBJECT_DETECTOR_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

FACE_LANDMARKER_PATH = get_resource_path("face_landmarker.task")
OBJECT_DETECTOR_PATH = get_resource_path("efficientdet_lite0.tflite")
