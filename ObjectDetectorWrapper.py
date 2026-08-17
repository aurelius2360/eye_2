import os
import urllib.request
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_resource_path(relative_path):
    """
    Get absolute path to resource, supporting PyInstaller bundles and dev environments.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class MediaPipeObjectDetector:
    """
    Wrapper for Google MediaPipe Object Detector task.
    Downloads the model automatically and runs detection to classify people and cell phones.
    """
    def __init__(self):
        model_path = get_resource_path('efficientdet_lite0.tflite')
        if not os.path.exists(model_path):
            print("Model file 'efficientdet_lite0.tflite' not found.")
            print("Downloading official efficientdet_lite0.tflite model (approx. 4.4 MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.45,  # 45% confidence threshold for reliability
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect_objects(self, rgb_frame):
        """
        Detects objects in the given RGB frame.
        Returns:
            person_count (int): Number of detected people
            phone_detected (bool): True if cell phone is detected
            detections (list): Raw MediaPipe detections
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self.detector.detect(mp_image)
        
        person_count = 0
        phone_detected = False
        
        for detection in res.detections:
            for category in detection.categories:
                name = category.category_name.lower()
                if name == "person":
                    person_count += 1
                elif name in ["cell phone", "phone", "mobile phone", "telephone"]:
                    phone_detected = True
                    
        return person_count, phone_detected, res.detections
