import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import urllib.request
from collections import deque
from ObjectDetectorWrapper import MediaPipeObjectDetector
from PreRollPostRollRecorder import start_full_exam_recording, trigger_cheat_recording, process_frame, run_end_of_exam_compression
from DeviceProctorScanner import DeviceProctorScanner
import sys

def get_resource_path(relative_path):
    """
    Get absolute path to resource, supporting PyInstaller bundles and dev environments.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class FaceMeshTracker:
    """
    Dual-compatibility wrapper for MediaPipe Face Mesh.
    Supports both the legacy mp.solutions API and the modern Tasks API.
    """
    def __init__(self):
        self.mode = "legacy"
        try:
            # Try importing legacy API
            import mediapipe.solutions.face_mesh as mp_face_mesh
            self.detector = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            print("MediaPipe: Initialized using legacy 'mp.solutions.face_mesh' API.")
        except (AttributeError, ModuleNotFoundError):
            # Fallback to modern Tasks API
            print("MediaPipe: Legacy API not available. Initializing modern Tasks API...")
            self.mode = "tasks"
            
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            model_path = get_resource_path('face_landmarker.task')
            if not os.path.exists(model_path):
                print(f"Model file '{model_path}' not found.")
                print("Downloading official face_landmarker.task model (approx. 5.6 MB)...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                try:
                    urllib.request.urlretrieve(url, model_path)
                    print("Download complete.")
                except Exception as e:
                    print(f"Error downloading model: {e}")
                    raise
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe: Initialized using modern 'FaceLandmarker' Tasks API.")

    def process(self, rgb_frame):
        """Processes the RGB frame and returns the first face's landmarks, or None."""
        if self.mode == "legacy":
            results = self.detector.process(rgb_frame)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = self.detector.detect(mp_image)
            if res.face_landmarks:
                return res.face_landmarks[0]
            return None

    def close(self):
        """Closes the underlying detector."""
        self.detector.close()


def crop_eye_region(frame, face_landmarks, eye_indices, padding_x=12, padding_y=8):
    """
    Crops the eye region from the frame based on landmark indices.
    Returns: cropped image, (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
    """
    img_h, img_w, _ = frame.shape
    
    pts = np.array([
        (face_landmarks[idx].x * img_w, face_landmarks[idx].y * img_h)
        for idx in eye_indices
    ])
    
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    
    crop_x_min = max(0, int(x_min - padding_x))
    crop_x_max = min(img_w, int(x_max + padding_x))
    crop_y_min = max(0, int(y_min - padding_y))
    crop_y_max = min(img_h, int(y_max + padding_y))
    
    cropped = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    return cropped, (crop_x_min, crop_y_min, crop_x_max, crop_y_max)


def draw_hud_panel(frame, title, x, y, w, h, bg_color=(20, 20, 20), border_color=(50, 50, 50)):
    """Draws a clean semi-transparent HUD panel container."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)
    
    # Title bar
    cv2.rectangle(frame, (x, y), (x + w, y + 25), border_color, -1)
    cv2.putText(frame, title, (x + 8, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
def draw_density_map(points_with_time, width=800, height=600, cheat_alarm=False, threshold_left_x=-0.12, threshold_right_x=0.12):
    """Draws a custom grid canvas plotting gaze points and the average gaze center."""
    gaze_map = np.zeros((height, width, 3), dtype=np.uint8)
    gaze_map[:] = (18, 18, 18) # Dark charcoal background
    
    # Draw dark grid lines
    grid_spacing = 50
    for x in range(0, width, grid_spacing):
        cv2.line(gaze_map, (x, 0), (x, height), (28, 28, 28), 1, cv2.LINE_AA)
    for y in range(0, height, grid_spacing):
        cv2.line(gaze_map, (0, y), (width, y), (28, 28, 28), 1, cv2.LINE_AA)
        
    # Draw center target axes (X and Y axes)
    cv2.line(gaze_map, (width // 2, 0), (width // 2, height), (80, 80, 80), 1, cv2.LINE_AA) # Y Axis
    cv2.line(gaze_map, (0, height // 2), (width, height // 2), (80, 80, 80), 1, cv2.LINE_AA) # X Axis
    
    # Label the X and Y axes
    cv2.putText(gaze_map, "Y-AXIS (CENTER)", (width // 2 + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(gaze_map, "X-AXIS (CENTER)", (20, height // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # Draw left and right monitor boundaries
    mx_left = int(width / 2 + threshold_left_x * 2500)
    mx_right = int(width / 2 + threshold_right_x * 2500)
    mx_left = max(0, min(width - 1, mx_left))
    mx_right = max(0, min(width - 1, mx_right))
    
    # Draw vertical red boundary lines to identify looking outside the monitor
    cv2.line(gaze_map, (mx_left, 0), (mx_left, height), (0, 0, 180), 2, cv2.LINE_AA)
    cv2.line(gaze_map, (mx_right, 0), (mx_right, height), (0, 0, 180), 2, cv2.LINE_AA)
    
    # Label the boundary lines
    cv2.putText(gaze_map, "LEFT MONITOR BOUNDARY", (mx_left + 10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1, cv2.LINE_AA)
    cv2.putText(gaze_map, "RIGHT MONITOR BOUNDARY", (mx_right - 180, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1, cv2.LINE_AA)
    
    # Extract points (x, y) from points_with_time where each element is (x, y, timestamp)
    points = [(p[0], p[1]) for p in points_with_time]
    
    if len(points) > 0:
        # Plot points with color fading from older (darker teal) to newer (bright cyan)
        num_pts = len(points)
        for i, (x, y) in enumerate(points):
            alpha = (i + 1) / num_pts
            # Fade color: older points are dark teal (80, 80, 0), newer points are bright cyan (255, 255, 0)
            color = (int(80 + 175 * alpha), int(80 + 175 * alpha), int(20 * alpha))
            radius = 3 if i < num_pts - 10 else 4
            cv2.circle(gaze_map, (x, y), radius, color, -1, cv2.LINE_AA)
            
        # Compute and draw average gaze point
        pts_arr = np.array(points)
        avg_x = int(np.mean(pts_arr[:, 0]))
        avg_y = int(np.mean(pts_arr[:, 1]))
        
        # Calculate standard deviation (spread)
        std_x = np.std(pts_arr[:, 0])
        std_y = np.std(pts_arr[:, 1])
        spread = math.hypot(std_x, std_y)
        
        # Draw average marker: orange/yellow dashed crosshair and target circle
        # Horizontal average line
        cv2.line(gaze_map, (0, avg_y), (width, avg_y), (0, 140, 255), 1, cv2.LINE_AA)
        # Vertical average line
        cv2.line(gaze_map, (avg_x, 0), (avg_x, height), (0, 140, 255), 1, cv2.LINE_AA)
        
        # Target circle representing average
        cv2.circle(gaze_map, (avg_x, avg_y), 8, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(gaze_map, (avg_x, avg_y), 14, (0, 165, 255), 2, cv2.LINE_AA)
        
        # Draw HUD info on density map
        cv2.putText(gaze_map, f"Gaze Points: {num_pts}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(gaze_map, f"Average Center: ({avg_x}, {avg_y})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(gaze_map, f"Gaze Spread (StdDev): {spread:.1f}px", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(gaze_map, "No Gaze Data - Look around to plot points", (width // 2 - 170, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
                    
    # Header title
    cv2.putText(gaze_map, "GAZE DENSITY MAP", (width - 170, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
    cv2.putText(gaze_map, "Press 'r' to Reset Map", (width - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
                
    # Draw Cheat Alarm indicator on Gaze Density Map if active
    if cheat_alarm:
        # Draw red border
        cv2.rectangle(gaze_map, (0, 0), (width, height), (0, 0, 255), 10)
        cv2.putText(gaze_map, "WARNING: LOOKING AWAY DETECTED", (width // 2 - 190, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                    
    return gaze_map


# Button click detection state variable
button_clicked = False

def mouse_callback(event, x, y, flags, param):
    global button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if coordinates fall within the button box [200, 240, 440, 290]
        if 200 <= x <= 440 and 240 <= y <= 290:
            button_clicked = True


def main():
    global button_clicked

    # Initialize the dual-compatibility tracker
    try:
        tracker = FaceMeshTracker()
    except Exception as e:
        print(f"Failed to initialize MediaPipe Face Tracker: {e}")
        return

    # Initialize the MediaPipe Object Detector
    try:
        object_detector = MediaPipeObjectDetector()
        print("MediaPipe Object Detector initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize MediaPipe Object Detector: {e}. Object proctoring disabled.")
        object_detector = None

    # Initialize BLE & Wi-Fi Device Proctoring Scanner (10-min / 600s audit interval)
    try:
        device_scanner = DeviceProctorScanner(audit_interval=600.0)
        print("Device Proctoring Scanner (BLE & Wi-Fi) initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Device Proctoring Scanner: {e}")
        device_scanner = None

    # Open webcam with index 1, fallback to 0
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera 1 not available. Falling back to Camera 0...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        tracker.close()
        return

    # Set frame resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Eye landmarks definitions
    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    LEFT_IRIS_CENTER = 468
    LEFT_IRIS_EDGE = 469
    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_EDGE = 474

    # Gaze trail history
    trail_length = 20
    gaze_trail = deque(maxlen=trail_length)

    # Gaze density map variables
    map_w, map_h = 800, 600
    gaze_map_points = deque(maxlen=500)
    
    # Initialize Gaze Density Map window
    cv2.namedWindow("Gaze Density Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Density Map", map_w, map_h)

    # Initialize and configure main window and mouse callback
    cv2.namedWindow("Eye & Pupil Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye & Pupil Tracker", 640, 480)
    cv2.setMouseCallback("Eye & Pupil Tracker", mouse_callback)
    cv2.setWindowProperty("Eye & Pupil Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Setup State Machine variables
    state = 'WELCOME'
    
    # Calibration retry configurations
    max_retries = 3
    center_retry_count = 0
    left_retry_count = 0
    right_retry_count = 0
    
    # Timings and temporary buffers for calibration
    stage_start_time = 0.0
    retry_delay_start = 0.0
    retry_next_state = 'WELCOME'
    retry_reason = ""
    
    calibration_offsets_x = []
    calibration_offsets_y = []
    left_calibration_offsets = []
    right_calibration_offsets = []

    # Calibrated biases and thresholds
    bias_x = 0.0
    bias_y = 0.0
    threshold_left_x = -0.10
    threshold_right_x = 0.10
    threshold_y = 0.10

    # Gaze state history for cheating detection: (calibrated_x, calibrated_y, timestamp, is_away)
    gaze_history = deque()
    cheat_alarm = False
    away_time = 0.0

    # Frame counter and object proctoring persistence variables
    frame_count = 0
    person_count = 1
    phone_detected = False
    detections = []
    obj_alarm_triggered = False
    obj_alarm_text = ""
    exam_recording_initialized = False

    # FPS Calculation
    prev_time = time.time()
    fps = 0.0

    print("\n---------------- Eye & Pupil Tracker Running ----------------")
    print("Calibration setup flow active. Follow onscreen prompts.")
    print("Press 'q' or 'ESC' to exit.")
    print("-------------------------------------------------------------\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        frame_count += 1

        # Flip horizontally for a mirror-like experience
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        
        # Prepare display canvas
        if state in ['WELCOME', 'CENTER_CALIB', 'LEFT_CALIB', 'RIGHT_CALIB', 'RETRY_DELAY']:
            display_frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            display_frame[:] = (18, 18, 18) # Dark charcoal background
        else:
            display_frame = frame.copy()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face landmarks
        face_landmarks = tracker.process(rgb_frame)

        # Draw a custom premium header dashboard
        cv2.rectangle(display_frame, (0, 0), (img_w, 55), (15, 15, 15), -1)
        cv2.line(display_frame, (0, 55), (img_w, 55), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "AI EYE & PUPIL MOTION TRACKER", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 240), 2, cv2.LINE_AA)

        # Trigger BLE & Wi-Fi baseline scan during setup flow if not completed/running
        if device_scanner and not device_scanner.get_status()['baseline_completed'] and not device_scanner.get_status()['is_scanning']:
            device_scanner.start_baseline_scan_async()

        # --- SETUP FLOW STATE MACHINE ---

        if state == 'WELCOME':
            # Display title and instructions
            cv2.putText(display_frame, "GAZE CALIBRATION SETUP", (img_w // 2 - 130, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 240), 2, cv2.LINE_AA)
            
            instructions = [
                "1. Keep your head still and look directly at the screen.",
                "2. Click start to calibrate your neutral center bias.",
                "3. Follow the target dot to the left and right edges.",
                "4. This maps screen boundaries to avoid false alarms."
            ]
            for idx, text in enumerate(instructions):
                cv2.putText(display_frame, text, (40, 150 + idx * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Draw green START button: [200, 240, 440, 290]
            btn_color = (0, 180, 100) # Emerald green
            cv2.rectangle(display_frame, (200, 240), (440, 290), btn_color, -1, cv2.LINE_AA)
            cv2.rectangle(display_frame, (200, 240), (440, 290), (0, 255, 150), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "START CALIBRATION", (225, 271),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Check keyboard trigger too (Spacebar)
            key = cv2.waitKey(1) & 0xFF
            if button_clicked or key == ord(' '):
                button_clicked = False
                state = 'CENTER_CALIB'
                stage_start_time = time.time()
                calibration_offsets_x.clear()
                calibration_offsets_y.clear()
                center_retry_count = 0
                print("Starting center calibration...")

        elif state == 'CENTER_CALIB':
            # Target dot in center (320, 240)
            cx, cy = img_w // 2, img_h // 2
            cv2.circle(display_frame, (cx, cy), 15, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)
            
            cv2.putText(display_frame, "LOOK DIRECTLY AT THE CENTER DOT", (img_w // 2 - 165, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "Keep your head perfectly still.", (img_w // 2 - 120, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
            
            progress = len(calibration_offsets_x) / 50.0
            cv2.putText(display_frame, f"Calibrating Center: {int(progress * 100)}%", (img_w // 2 - 100, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Progress bar
            bar_w = 200
            bar_x = img_w // 2 - bar_w // 2
            cv2.rectangle(display_frame, (bar_x, 380), (bar_x + bar_w, 390), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (bar_x, 380), (bar_x + int(bar_w * progress), 390), (0, 255, 0), -1)

            # Process landmarks
            if face_landmarks:
                def to_px(lm):
                    return int(lm.x * img_w), int(lm.y * img_h)

                l_contour_pts = [to_px(face_landmarks[idx]) for idx in LEFT_EYE_CONTOUR]
                r_contour_pts = [to_px(face_landmarks[idx]) for idx in RIGHT_EYE_CONTOUR]
                
                l_iris_center = to_px(face_landmarks[LEFT_IRIS_CENTER])
                r_iris_center = to_px(face_landmarks[RIGHT_IRIS_CENTER])

                l_eye_center = np.mean(l_contour_pts, axis=0)
                r_eye_center = np.mean(r_contour_pts, axis=0)

                l_outer = to_px(face_landmarks[33])
                l_inner = to_px(face_landmarks[133])
                l_width = max(1.0, math.hypot(l_outer[0] - l_inner[0], l_outer[1] - l_inner[1]))
                
                r_inner = to_px(face_landmarks[362])
                r_outer = to_px(face_landmarks[263])
                r_width = max(1.0, math.hypot(r_outer[0] - r_inner[0], r_outer[1] - r_inner[1]))
                
                l_top = to_px(face_landmarks[159])
                l_bot = to_px(face_landmarks[145])
                l_height = max(1.0, math.hypot(l_top[0] - l_bot[0], l_top[1] - l_bot[1]))
                
                r_top = to_px(face_landmarks[386])
                r_bot = to_px(face_landmarks[374])
                r_height = max(1.0, math.hypot(r_top[0] - r_bot[0], r_top[1] - r_bot[1]))

                ear_l = l_height / l_width
                ear_r = r_height / r_width
                avg_ear = (ear_l + ear_r) / 2.0
                blink = avg_ear < 0.16

                if not blink:
                    l_offset_x = (l_iris_center[0] - l_eye_center[0]) / l_width
                    l_offset_y = (l_iris_center[1] - l_eye_center[1]) / l_height
                    r_offset_x = (r_iris_center[0] - r_eye_center[0]) / r_width
                    r_offset_y = (r_iris_center[1] - r_eye_center[1]) / r_height

                    avg_offset_x = (l_offset_x + r_offset_x) / 2.0
                    avg_offset_y = (l_offset_y + r_offset_y) / 2.0

                    calibration_offsets_x.append(avg_offset_x)
                    calibration_offsets_y.append(avg_offset_y)

            # Check if 50 frames collected
            if len(calibration_offsets_x) >= 50:
                std_x = np.std(calibration_offsets_x)
                std_y = np.std(calibration_offsets_y)
                
                # Check for high variance (user moving head too much)
                if std_x > 0.035 or std_y > 0.035:
                    center_retry_count += 1
                    if center_retry_count < max_retries:
                        state = 'RETRY_DELAY'
                        retry_next_state = 'CENTER_CALIB'
                        retry_reason = "Too much movement detected. Keep still!"
                        retry_delay_start = time.time()
                    else:
                        bias_x = 0.0
                        bias_y = 0.0
                        state = 'RUNNING'
                        cv2.setWindowProperty("Eye & Pupil Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Eye & Pupil Tracker", 640, 480)
                        print("Center calibration failed 3 times. Using default (0, 0) center bias.")
                else:
                    bias_x = np.mean(calibration_offsets_x)
                    bias_y = np.mean(calibration_offsets_y)
                    state = 'RUNNING'
                    cv2.setWindowProperty("Eye & Pupil Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Eye & Pupil Tracker", 640, 480)
                    print(f"Center calibration succeeded. Bias: X={bias_x:.4f}, Y={bias_y:.4f}")
            
            # Timeout check (8 seconds)
            elif time.time() - stage_start_time > 8.0:
                center_retry_count += 1
                if center_retry_count < max_retries:
                    state = 'RETRY_DELAY'
                    retry_next_state = 'CENTER_CALIB'
                    retry_reason = "Timeout: Face not detected or eyes closed."
                    retry_delay_start = time.time()
                else:
                    bias_x = 0.0
                    bias_y = 0.0
                    state = 'RUNNING'
                    cv2.setWindowProperty("Eye & Pupil Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Eye & Pupil Tracker", 640, 480)
                    print("Center calibration failed 3 times due to timeout. Using default center (0, 0).")

        elif state == 'LEFT_CALIB':
            # Target dot on the left edge: x=64 (10% of 640), y=240
            cx, cy = int(img_w * 0.1), img_h // 2
            cv2.circle(display_frame, (cx, cy), 15, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)
            
            cv2.putText(display_frame, "LOOK AT THE LEFT DOT", (img_w // 2 - 110, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
            
            elapsed = time.time() - stage_start_time
            countdown = max(0.0, 2.0 - elapsed)
            cv2.putText(display_frame, f"Recording: {countdown:.1f}s", (img_w // 2 - 60, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if face_landmarks:
                def to_px(lm):
                    return int(lm.x * img_w), int(lm.y * img_h)

                l_contour_pts = [to_px(face_landmarks[idx]) for idx in LEFT_EYE_CONTOUR]
                r_contour_pts = [to_px(face_landmarks[idx]) for idx in RIGHT_EYE_CONTOUR]
                
                l_iris_center = to_px(face_landmarks[LEFT_IRIS_CENTER])
                r_iris_center = to_px(face_landmarks[RIGHT_IRIS_CENTER])

                l_eye_center = np.mean(l_contour_pts, axis=0)
                r_eye_center = np.mean(r_contour_pts, axis=0)

                l_outer = to_px(face_landmarks[33])
                l_inner = to_px(face_landmarks[133])
                l_width = max(1.0, math.hypot(l_outer[0] - l_inner[0], l_outer[1] - l_inner[1]))
                
                r_inner = to_px(face_landmarks[362])
                r_outer = to_px(face_landmarks[263])
                r_width = max(1.0, math.hypot(r_outer[0] - r_inner[0], r_outer[1] - r_inner[1]))
                
                l_top = to_px(face_landmarks[159])
                l_bot = to_px(face_landmarks[145])
                l_height = max(1.0, math.hypot(l_top[0] - l_bot[0], l_top[1] - l_bot[1]))
                
                r_top = to_px(face_landmarks[386])
                r_bot = to_px(face_landmarks[374])
                r_height = max(1.0, math.hypot(r_top[0] - r_bot[0], r_top[1] - r_bot[1]))

                ear_l = l_height / l_width
                ear_r = r_height / r_width
                avg_ear = (ear_l + ear_r) / 2.0
                blink = avg_ear < 0.16

                if not blink:
                    l_offset_x = (l_iris_center[0] - l_eye_center[0]) / l_width
                    r_offset_x = (r_iris_center[0] - r_eye_center[0]) / r_width
                    avg_offset_x = (l_offset_x + r_offset_x) / 2.0
                    
                    calibrated_x = avg_offset_x - bias_x
                    left_calibration_offsets.append(calibrated_x)

            if elapsed >= 2.0:
                if len(left_calibration_offsets) < 15:
                    left_retry_count += 1
                    if left_retry_count < max_retries:
                        state = 'RETRY_DELAY'
                        retry_next_state = 'LEFT_CALIB'
                        retry_reason = "Face not tracked enough. Look at the LEFT dot!"
                        retry_delay_start = time.time()
                    else:
                        threshold_left_x = -0.12
                        state = 'RIGHT_CALIB'
                        stage_start_time = time.time()
                        right_calibration_offsets.clear()
                        right_retry_count = 0
                        print("Left calibration failed 3 times. Using default -0.12.")
                else:
                    left_limit = np.mean(left_calibration_offsets)
                    # Check if offset is negative as expected when looking left
                    if left_limit >= -0.04:
                        left_retry_count += 1
                        if left_retry_count < max_retries:
                            state = 'RETRY_DELAY'
                            retry_next_state = 'LEFT_CALIB'
                            retry_reason = "Did not look left. Keep eyes on the LEFT dot!"
                            retry_delay_start = time.time()
                        else:
                            threshold_left_x = -0.12
                            state = 'RIGHT_CALIB'
                            stage_start_time = time.time()
                            right_calibration_offsets.clear()
                            right_retry_count = 0
                            print("Left calibration failed 3 times (invalid offset). Using default -0.12.")
                    else:
                        threshold_left_x = left_limit * 0.8
                        state = 'RIGHT_CALIB'
                        stage_start_time = time.time()
                        right_calibration_offsets.clear()
                        right_retry_count = 0
                        print(f"Left calibration succeeded. Threshold: {threshold_left_x:.4f} (limit={left_limit:.4f})")

        elif state == 'RIGHT_CALIB':
            # Target dot on the right edge: x=576 (90% of 640), y=240
            cx, cy = int(img_w * 0.9), img_h // 2
            cv2.circle(display_frame, (cx, cy), 15, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)
            
            cv2.putText(display_frame, "LOOK AT THE RIGHT DOT", (img_w // 2 - 115, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
            
            elapsed = time.time() - stage_start_time
            countdown = max(0.0, 2.0 - elapsed)
            cv2.putText(display_frame, f"Recording: {countdown:.1f}s", (img_w // 2 - 60, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if face_landmarks:
                def to_px(lm):
                    return int(lm.x * img_w), int(lm.y * img_h)

                l_contour_pts = [to_px(face_landmarks[idx]) for idx in LEFT_EYE_CONTOUR]
                r_contour_pts = [to_px(face_landmarks[idx]) for idx in RIGHT_EYE_CONTOUR]
                
                l_iris_center = to_px(face_landmarks[LEFT_IRIS_CENTER])
                r_iris_center = to_px(face_landmarks[RIGHT_IRIS_CENTER])

                l_eye_center = np.mean(l_contour_pts, axis=0)
                r_eye_center = np.mean(r_contour_pts, axis=0)

                l_outer = to_px(face_landmarks[33])
                l_inner = to_px(face_landmarks[133])
                l_width = max(1.0, math.hypot(l_outer[0] - l_inner[0], l_outer[1] - l_inner[1]))
                
                r_inner = to_px(face_landmarks[362])
                r_outer = to_px(face_landmarks[263])
                r_width = max(1.0, math.hypot(r_outer[0] - r_inner[0], r_outer[1] - r_inner[1]))
                
                l_top = to_px(face_landmarks[159])
                l_bot = to_px(face_landmarks[145])
                l_height = max(1.0, math.hypot(l_top[0] - l_bot[0], l_top[1] - l_bot[1]))
                
                r_top = to_px(face_landmarks[386])
                r_bot = to_px(face_landmarks[374])
                r_height = max(1.0, math.hypot(r_top[0] - r_bot[0], r_top[1] - r_bot[1]))

                ear_l = l_height / l_width
                ear_r = r_height / r_width
                avg_ear = (ear_l + ear_r) / 2.0
                blink = avg_ear < 0.16

                if not blink:
                    l_offset_x = (l_iris_center[0] - l_eye_center[0]) / l_width
                    r_offset_x = (r_iris_center[0] - r_eye_center[0]) / r_width
                    avg_offset_x = (l_offset_x + r_offset_x) / 2.0
                    
                    calibrated_x = avg_offset_x - bias_x
                    right_calibration_offsets.append(calibrated_x)

            if elapsed >= 2.0:
                if len(right_calibration_offsets) < 15:
                    right_retry_count += 1
                    if right_retry_count < max_retries:
                        state = 'RETRY_DELAY'
                        retry_next_state = 'RIGHT_CALIB'
                        retry_reason = "Face not tracked enough. Look at the RIGHT dot!"
                        retry_delay_start = time.time()
                    else:
                        threshold_right_x = 0.12
                        state = 'RUNNING'
                        print("Right calibration failed 3 times. Using default 0.12.")
                else:
                    right_limit = np.mean(right_calibration_offsets)
                    # Check if offset is positive as expected when looking right
                    if right_limit <= 0.04:
                        right_retry_count += 1
                        if right_retry_count < max_retries:
                            state = 'RETRY_DELAY'
                            retry_next_state = 'RIGHT_CALIB'
                            retry_reason = "Did not look right. Keep eyes on the RIGHT dot!"
                            retry_delay_start = time.time()
                        else:
                            threshold_right_x = 0.12
                            state = 'RUNNING'
                            print("Right calibration failed 3 times (invalid offset). Using default 0.12.")
                    else:
                        threshold_right_x = right_limit * 0.8
                        state = 'RUNNING'
                        print(f"Right calibration succeeded. Threshold: {threshold_right_x:.4f} (limit={right_limit:.4f})")

        elif state == 'RETRY_DELAY':
            # Display failure overlay and warning text
            cv2.putText(display_frame, "CALIBRATION ATTEMPT FAILED", (img_w // 2 - 170, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show retry reason
            cv2.putText(display_frame, retry_reason, (img_w // 2 - len(retry_reason)*4, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Show countdown
            rem = max(0.0, 2.5 - (time.time() - retry_delay_start))
            cv2.putText(display_frame, f"Restarting in {rem:.1f} seconds...", (img_w // 2 - 110, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
            
            # Attempt numbers mapping
            att_num = 0
            if retry_next_state == 'CENTER_CALIB':
                att_num = center_retry_count + 1
            elif retry_next_state == 'LEFT_CALIB':
                att_num = left_retry_count + 1
            elif retry_next_state == 'RIGHT_CALIB':
                att_num = right_retry_count + 1
                
            cv2.putText(display_frame, f"(Attempt {att_num} of {max_retries})", (img_w // 2 - 70, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv2.LINE_AA)
            
            if time.time() - retry_delay_start >= 2.5:
                # Clear buffers and transition back
                if retry_next_state == 'CENTER_CALIB':
                    calibration_offsets_x.clear()
                    calibration_offsets_y.clear()
                elif retry_next_state == 'LEFT_CALIB':
                    left_calibration_offsets.clear()
                elif retry_next_state == 'RIGHT_CALIB':
                    right_calibration_offsets.clear()
                
                state = retry_next_state
                stage_start_time = time.time()

        elif state == 'RUNNING':
            # Initialize Type 1 continuous recording once when we enter RUNNING state
            if not exam_recording_initialized:
                start_full_exam_recording(img_w, img_h)
                exam_recording_initialized = True

            # --- RUNNING TRACKER LOGIC ---
            if face_landmarks:
                def to_px(lm):
                    return int(lm.x * img_w), int(lm.y * img_h)

                l_contour_pts = [to_px(face_landmarks[idx]) for idx in LEFT_EYE_CONTOUR]
                r_contour_pts = [to_px(face_landmarks[idx]) for idx in RIGHT_EYE_CONTOUR]
                
                l_iris_center = to_px(face_landmarks[LEFT_IRIS_CENTER])
                l_iris_edge = to_px(face_landmarks[LEFT_IRIS_EDGE])
                r_iris_center = to_px(face_landmarks[RIGHT_IRIS_CENTER])
                r_iris_edge = to_px(face_landmarks[RIGHT_IRIS_EDGE])

                # Draw Eyelid contours as Cyan dots on main frame
                for pt in l_contour_pts + r_contour_pts:
                    cv2.circle(display_frame, pt, 2, (255, 255, 0), -1, cv2.LINE_AA)

                # Calculate pupil/iris radii
                l_radius = int(math.hypot(l_iris_center[0] - l_iris_edge[0], l_iris_center[1] - l_iris_edge[1]))
                r_radius = int(math.hypot(r_iris_center[0] - r_iris_edge[0], r_iris_center[1] - r_iris_edge[1]))
                l_radius = max(1, l_radius)
                r_radius = max(1, r_radius)

                # Draw precise Magenta circle around pupil/iris on main frame
                cv2.circle(display_frame, l_iris_center, l_radius, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(display_frame, r_iris_center, r_radius, (255, 0, 255), 1, cv2.LINE_AA)
                
                # Draw Lime green pupil center dots on main frame
                cv2.circle(display_frame, l_iris_center, 2, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, r_iris_center, 2, (0, 255, 0), -1, cv2.LINE_AA)

                # --- Calculate Pupil Displacement ---
                l_eye_center = np.mean(l_contour_pts, axis=0)
                r_eye_center = np.mean(r_contour_pts, axis=0)

                l_outer = to_px(face_landmarks[33])
                l_inner = to_px(face_landmarks[133])
                l_width = max(1.0, math.hypot(l_outer[0] - l_inner[0], l_outer[1] - l_inner[1]))
                
                r_inner = to_px(face_landmarks[362])
                r_outer = to_px(face_landmarks[263])
                r_width = max(1.0, math.hypot(r_outer[0] - r_inner[0], r_outer[1] - r_inner[1]))
                
                l_top = to_px(face_landmarks[159])
                l_bot = to_px(face_landmarks[145])
                l_height = max(1.0, math.hypot(l_top[0] - l_bot[0], l_top[1] - l_bot[1]))
                
                r_top = to_px(face_landmarks[386])
                r_bot = to_px(face_landmarks[374])
                r_height = max(1.0, math.hypot(r_top[0] - r_bot[0], r_top[1] - r_bot[1]))

                # Blink Detection using EAR
                ear_l = l_height / l_width
                ear_r = r_height / r_width
                avg_ear = (ear_l + ear_r) / 2.0
                blink = avg_ear < 0.16

                # Normalized pupil offsets
                l_offset_x = (l_iris_center[0] - l_eye_center[0]) / l_width
                l_offset_y = (l_iris_center[1] - l_eye_center[1]) / l_height
                r_offset_x = (r_iris_center[0] - r_eye_center[0]) / r_width
                r_offset_y = (r_iris_center[1] - r_eye_center[1]) / r_height

                avg_offset_x = (l_offset_x + r_offset_x) / 2.0
                avg_offset_y = (l_offset_y + r_offset_y) / 2.0

                # Apply calibration bias offsets
                calibrated_x = avg_offset_x - bias_x
                calibrated_y = avg_offset_y - bias_y
                
                # Append to gaze trail for radar HUD
                gaze_trail.append((calibrated_x, calibrated_y))

                # Append to density map points with timestamp if not blinking
                if not blink:
                    mx = int(map_w / 2 + calibrated_x * 2500)
                    my = int(map_h / 2 + calibrated_y * 2500)
                    mx = max(0, min(map_w - 1, mx))
                    my = max(0, min(map_h - 1, my))
                    gaze_map_points.append((mx, my, time.time()))

                # Determine if user is looking away (Raw Math Thresholding)
                is_away = False
                gaze_state = "Center"
                if calibrated_x > threshold_right_x:
                    gaze_state = "Looking Right"
                    is_away = True
                elif calibrated_x < threshold_left_x:
                    gaze_state = "Looking Left"
                    is_away = True
                
                if calibrated_y > threshold_y:
                    gaze_state += " / Down" if gaze_state != "Center" else "Looking Down"
                    is_away = True
                elif calibrated_y < -threshold_y:
                    gaze_state += " / Up" if gaze_state != "Center" else "Looking Up"
                    is_away = True

                # Store event in sliding window gaze history (not during blink)
                if not blink:
                    gaze_history.append((calibrated_x, calibrated_y, time.time(), is_away))

                # Display current gaze and blink state
                state_text = "BLINKING" if blink else gaze_state.upper()
                state_color = (0, 0, 255) if blink else ((0, 255, 0) if gaze_state == "Center" else (255, 120, 0))
                
                cv2.putText(display_frame, f"GAZE STATE: {state_text}", (img_w // 2 - 100, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)

                # --- Draw Picture-in-Picture (PIP) Zoom Windows ---
                l_crop, l_box = crop_eye_region(frame, face_landmarks, LEFT_EYE_CONTOUR)
                if l_crop.size > 0:
                    l_zoom = cv2.resize(l_crop, (120, 80))
                    lc_w, lc_h = 120, 80
                    crop_box_w = l_box[2] - l_box[0]
                    crop_box_h = l_box[3] - l_box[1]
                    if crop_box_w > 0 and crop_box_h > 0:
                        mapped_center_x = int((l_iris_center[0] - l_box[0]) * (lc_w / crop_box_w))
                        mapped_center_y = int((l_iris_center[1] - l_box[1]) * (lc_h / crop_box_h))
                        mapped_radius = int(l_radius * (lc_w / crop_box_w))
                        cv2.circle(l_zoom, (mapped_center_x, mapped_center_y), mapped_radius, (255, 0, 255), 1, cv2.LINE_AA)
                        cv2.drawMarker(l_zoom, (mapped_center_x, mapped_center_y), (0, 255, 0), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
                    
                    draw_hud_panel(display_frame, "LEFT EYE ZOOM", 15, 70, 130, 105)
                    display_frame[90:170, 20:140] = l_zoom

                r_crop, r_box = crop_eye_region(frame, face_landmarks, RIGHT_EYE_CONTOUR)
                if r_crop.size > 0:
                    r_zoom = cv2.resize(r_crop, (120, 80))
                    rc_w, rc_h = 120, 80
                    crop_box_w = r_box[2] - r_box[0]
                    crop_box_h = r_box[3] - r_box[1]
                    if crop_box_w > 0 and crop_box_h > 0:
                        mapped_center_x = int((r_iris_center[0] - r_box[0]) * (rc_w / crop_box_w))
                        mapped_center_y = int((r_iris_center[1] - r_box[1]) * (rc_h / crop_box_h))
                        mapped_radius = int(r_radius * (rc_w / crop_box_w))
                        cv2.circle(r_zoom, (mapped_center_x, mapped_center_y), mapped_radius, (255, 0, 255), 1, cv2.LINE_AA)
                        cv2.drawMarker(r_zoom, (mapped_center_x, mapped_center_y), (0, 255, 0), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

                    draw_hud_panel(display_frame, "RIGHT EYE ZOOM", img_w - 145, 70, 130, 105)
                    display_frame[90:170, img_w - 140:img_w - 20] = r_zoom

                # --- Draw Gaze Radar HUD ---
                radar_x = 15
                radar_y = img_h - 130
                radar_w = 110
                radar_h = 110
                radar_center = (radar_x + radar_w // 2, radar_y + radar_h // 2 + 10)
                radar_max_r = 40

                draw_hud_panel(display_frame, "GAZE RADAR", radar_x, radar_y, radar_w, radar_h)
                cv2.line(display_frame, (radar_center[0] - radar_max_r, radar_center[1]), (radar_center[0] + radar_max_r, radar_center[1]), (60, 60, 60), 1, cv2.LINE_AA)
                cv2.line(display_frame, (radar_center[0], radar_center[1] - radar_max_r), (radar_center[0], radar_center[1] + radar_max_r), (60, 60, 60), 1, cv2.LINE_AA)
                cv2.circle(display_frame, radar_center, radar_max_r, (80, 80, 80), 1, cv2.LINE_AA)
                cv2.circle(display_frame, radar_center, radar_max_r // 2, (50, 50, 50), 1, cv2.LINE_AA)

                if len(gaze_trail) > 0:
                    scale_factor = 250.0
                    for i, (gx, gy) in enumerate(list(gaze_trail)):
                        px = int(radar_center[0] + gx * scale_factor)
                        py = int(radar_center[1] + gy * scale_factor)
                        dist_from_c = math.hypot(px - radar_center[0], py - radar_center[1])
                        if dist_from_c > radar_max_r:
                            angle = math.atan2(py - radar_center[1], px - radar_center[0])
                            px = int(radar_center[0] + radar_max_r * math.cos(angle))
                            py = int(radar_center[1] + radar_max_r * math.sin(angle))
                        
                        alpha = (i + 1) / len(gaze_trail)
                        color = (int(0 * alpha), int(255 * alpha), int(240 * alpha))
                        pt_size = max(1, int(4 * alpha))
                        cv2.circle(display_frame, (px, py), pt_size, color, -1, cv2.LINE_AA)
                    
                    curr_x, curr_y = gaze_trail[-1]
                    px = int(radar_center[0] + curr_x * scale_factor)
                    py = int(radar_center[1] + gy * scale_factor)
                    dist_from_c = math.hypot(px - radar_center[0], py - radar_center[1])
                    if dist_from_c > radar_max_r:
                        angle = math.atan2(py - radar_center[1], px - radar_center[0])
                        px = int(radar_center[0] + radar_max_r * math.cos(angle))
                        py = int(radar_center[1] + radar_max_r * math.sin(angle))
                    
                    cv2.circle(display_frame, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(display_frame, (px, py), 6, (0, 0, 255), 1, cv2.LINE_AA)

            else:
                # No face detected
                cv2.putText(display_frame, "NO FACE DETECTED", (img_w // 2 - 80, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            # --- Stopwatch Gaze Durations & Cheat Alarm ---
            now = time.time()
            # Prune points older than 5 seconds from sliding window
            while gaze_history and now - gaze_history[0][2] > 5.0:
                gaze_history.popleft()
                
            # Prune density map points
            while gaze_map_points and now - gaze_map_points[0][2] > 5.0:
                gaze_map_points.popleft()

            # Calculate total time looking away in the last 5 seconds
            away_time = 0.0
            if len(gaze_history) > 0:
                for i in range(len(gaze_history) - 1):
                    dt = gaze_history[i+1][2] - gaze_history[i][2]
                    if gaze_history[i][3]:  # was looking away
                        away_time += dt
                # Add time since last point to 'now' if last point was looking away
                if gaze_history[-1][3]:
                    away_time += (now - gaze_history[-1][2])

            # --- Object Detection Proctoring Alerts (Runs every 15 frames) ---
            if object_detector and (frame_count % 15 == 0):
                try:
                    person_count, phone_detected, detections = object_detector.detect_objects(rgb_frame)
                    
                    obj_alarm_triggered = False
                    obj_alarm_text = ""
                    
                    if person_count == 0:
                        print("ALARM: User left the camera!")
                        obj_alarm_triggered = True
                        obj_alarm_text = "ALARM: NO USER DETECTED"
                    elif person_count > 1:
                        print("ALARM: Multiple people in the room!")
                        obj_alarm_triggered = True
                        obj_alarm_text = "ALARM: MULTIPLE PEOPLE DETECTED"
                    elif phone_detected:
                        print("ALARM: Cell phone detected!")
                        obj_alarm_triggered = True
                        obj_alarm_text = "ALARM: CELL PHONE DETECTED"
                except Exception as e:
                    print(f"Error running object detector: {e}")

            # Draw red bounding boxes for violating detections (from last active run)
            if object_detector:
                for detection in detections:
                    category = detection.categories[0]
                    label = category.category_name.lower()
                    score = category.score
                    
                    if label in ["person", "cell phone"]:
                        bbox = detection.bounding_box
                        bx, by, bw, bh = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                        bx = max(0, min(img_w - 1, bx))
                        by = max(0, min(img_h - 1, by))
                        bw = max(1, min(img_w - bx, bw))
                        bh = max(1, min(img_h - by, bh))
                        
                        color = (0, 0, 255) # Red bounding box
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)
                        cv2.putText(display_frame, f"{label.upper()} ({score:.2f})", (bx + 5, by + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # --- BLE & Wi-Fi Periodic Device Proctoring (10-minute audit) ---
            dev_alarm_triggered = False
            dev_alarm_text = ""
            if device_scanner:
                device_scanner.check_and_trigger_periodic_audit(now)
                dev_status = device_scanner.get_status()
                dev_alarm_triggered = dev_status['alarm_triggered']
                if dev_alarm_triggered:
                    dev_alarm_text = dev_status['alarm_message']

                # Display device scanner HUD info on bottom left overlay panel
                scan_str = "Scanning..." if dev_status['is_scanning'] else f"BLE:{dev_status['ble_count']} Wi-Fi:{dev_status['wifi_count']}"
                cv2.putText(display_frame, f"NET: {scan_str}", (15, img_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 240), 1, cv2.LINE_AA)

            # Trigger Alarm if away > 3s OR object detector alarm OR device scanner alarm triggered
            cheat_alarm = (away_time > 3.0) or obj_alarm_triggered or dev_alarm_triggered
            if cheat_alarm:
                trigger_cheat_recording(img_w, img_h)

            # Render Cheat Alarm overlay on main window
            if cheat_alarm:
                # Red translucent warning overlay
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (img_w, img_h), (0, 0, 100), -1)
                cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)
                
                # Draw alarm UI box and text
                cv2.rectangle(display_frame, (img_w // 2 - 240, img_h // 2 - 40), (img_w // 2 + 240, img_h // 2 + 40), (0, 0, 200), -1, cv2.LINE_AA)
                if obj_alarm_triggered:
                    cv2.putText(display_frame, obj_alarm_text, (img_w // 2 - len(obj_alarm_text)*6, img_h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, "PROCTORING VIOLATION", (img_w // 2 - 95, img_h // 2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
                elif dev_alarm_triggered:
                    cv2.putText(display_frame, dev_alarm_text, (img_w // 2 - len(dev_alarm_text)*5, img_h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, "UNAUTHORIZED DEVICE DETECTED", (img_w // 2 - 130, img_h // 2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(display_frame, "WARNING: LOOKING AWAY DETECTED", (img_w // 2 - 180, img_h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Away: {away_time:.1f}s of last 5s (Max 3.0s)", (img_w // 2 - 165, img_h // 2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
            elif away_time > 0.5:
                # Show minor warning gauge if looking away but hasn't reached 3.0s yet
                cv2.putText(display_frame, f"Gaze Alert: {away_time:.1f}s / 3.0s", (img_w // 2 - 80, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        # Calculate FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        prev_time = curr_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        
        # Display FPS in header
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (img_w - 110, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Process the frame for pre-roll/post-roll and full exam recording
        if exam_recording_initialized:
            process_frame(display_frame)

        # Show main tracker frame
        cv2.imshow("Eye & Pupil Tracker", display_frame)

        # Draw and show Gaze Density Map frame
        if state == 'RUNNING':
            gaze_map_img = draw_density_map(gaze_map_points, map_w, map_h, cheat_alarm=cheat_alarm,
                                            threshold_left_x=threshold_left_x, threshold_right_x=threshold_right_x)
            cv2.imshow("Gaze Density Map", gaze_map_img)
        else:
            empty_map = draw_density_map(deque(), map_w, map_h, cheat_alarm=False)
            cv2.imshow("Gaze Density Map", empty_map)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC is 27
            # End of test check: capture final 10 seconds pre-roll if alarm is active
            if cheat_alarm:
                trigger_cheat_recording(img_w, img_h, is_test_ending=True)
            break
        elif key == ord('c'):
            # Recalibrate center - Reset state and variables
            state = 'WELCOME'
            button_clicked = False
            cv2.setWindowProperty("Eye & Pupil Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            gaze_map_points.clear()
            gaze_history.clear()
            calibration_offsets_x.clear()
            calibration_offsets_y.clear()
            center_retry_count = 0
            cheat_alarm = False
            print("Resetting tracker to Welcome screen for center calibration...")
        elif key == ord('r'):
            gaze_map_points.clear()
            gaze_history.clear()
            cheat_alarm = False
            print("Gaze maps cleared.")

    # Release camera and close GUI windows immediately so they disappear from the screen
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    # Run end of exam compression for all Type 1 (Full Exam) and Type 2 (Anomaly Clips) videos
    if exam_recording_initialized:
        print("[System] Compressing exam videos, please wait...")
        compressed_paths = run_end_of_exam_compression()
        print(f"[System] End-of-exam compression finished. Optimized WebM files: {compressed_paths}")

    print("Tracker shutdown successfully.")

if __name__ == "__main__":
    main()
