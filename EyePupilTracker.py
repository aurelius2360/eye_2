import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import datetime
import os
import sys
from collections import deque

from model_utils import (
    ensure_model_file,
    FACE_LANDMARKER_URL,
    OBJECT_DETECTOR_URL,
    FACE_LANDMARKER_PATH,
    OBJECT_DETECTOR_PATH
)
from PreRollPostRollRecorder import (
    load_config,
    start_full_exam_recording,
    trigger_cheat_recording,
    capture_violation_snapshot,
    record_violation_event,
    process_frame,
    run_end_of_exam_compression,
    update_live_telemetry,
    save_live_frames
)
from DeviceProctorScanner import DeviceProctorScanner

# Eye & Iris Landmark Indices
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_IRIS_CENTER = 468
LEFT_IRIS_EDGE = 469
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_EDGE = 474


class FaceMeshTracker:
    """MediaPipe Face Landmarker tracker wrapper."""
    def __init__(self):
        model_path = ensure_model_file(FACE_LANDMARKER_PATH, FACE_LANDMARKER_URL)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        """Processes RGB frame and returns first face landmarks, or None."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self.detector.detect(mp_image)
        return res.face_landmarks[0] if res.face_landmarks else None

    def close(self):
        self.detector.close()


class MediaPipeObjectDetector:
    """MediaPipe Object Detector for classifying multiple people and mobile phones."""
    def __init__(self):
        model_path = ensure_model_file(OBJECT_DETECTOR_PATH, OBJECT_DETECTOR_URL)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.45,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect_objects(self, rgb_frame):
        """Detects objects in RGB frame and returns (person_count, phone_detected, detections)."""
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


def extract_eye_metrics(face_landmarks, img_w, img_h):
    """
    Unified extraction helper for Eye Aspect Ratio (EAR), blink detection,
    pupil centers, radii, and normalized offsets. Single source of truth.
    """
    def to_px(lm):
        return int(lm.x * img_w), int(lm.y * img_h)

    l_contour_pts = [to_px(face_landmarks[idx]) for idx in LEFT_EYE_CONTOUR]
    r_contour_pts = [to_px(face_landmarks[idx]) for idx in RIGHT_EYE_CONTOUR]

    l_iris_center = to_px(face_landmarks[LEFT_IRIS_CENTER])
    l_iris_edge = to_px(face_landmarks[LEFT_IRIS_EDGE])
    r_iris_center = to_px(face_landmarks[RIGHT_IRIS_CENTER])
    r_iris_edge = to_px(face_landmarks[RIGHT_IRIS_EDGE])

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

    l_radius = max(1, int(math.hypot(l_iris_center[0] - l_iris_edge[0], l_iris_center[1] - l_iris_edge[1])))
    r_radius = max(1, int(math.hypot(r_iris_center[0] - r_iris_edge[0], r_iris_center[1] - r_iris_edge[1])))

    l_offset_x = (l_iris_center[0] - l_eye_center[0]) / l_width
    l_offset_y = (l_iris_center[1] - l_eye_center[1]) / l_height
    r_offset_x = (r_iris_center[0] - r_eye_center[0]) / r_width
    r_offset_y = (r_iris_center[1] - r_eye_center[1]) / r_height

    raw_offset_x = (l_offset_x + r_offset_x) / 2.0
    raw_offset_y = (l_offset_y + r_offset_y) / 2.0

    return {
        "l_contour_pts": l_contour_pts,
        "r_contour_pts": r_contour_pts,
        "l_iris_center": l_iris_center,
        "r_iris_center": r_iris_center,
        "l_radius": l_radius,
        "r_radius": r_radius,
        "avg_ear": avg_ear,
        "blink": blink,
        "raw_offset_x": raw_offset_x,
        "raw_offset_y": raw_offset_y
    }


def draw_density_map(points_with_time, width=800, height=600, cheat_alarm=False,
                     threshold_left_x=-0.12, threshold_right_x=0.12, threshold_y=0.09):
    """
    Renders high-resolution Gaze Density Map & Analysis Canvas for the Admin Portal.
    """
    gaze_map = np.zeros((height, width, 3), dtype=np.uint8)
    gaze_map[:] = (18, 18, 18) # Dark charcoal background

    # Grid lines
    for x in range(0, width, 50):
        cv2.line(gaze_map, (x, 0), (x, height), (28, 28, 28), 1, cv2.LINE_AA)
    for y in range(0, height, 50):
        cv2.line(gaze_map, (0, y), (width, y), (28, 28, 28), 1, cv2.LINE_AA)

    # Target center crosshairs
    cv2.line(gaze_map, (width // 2, 0), (width // 2, height), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(gaze_map, (0, height // 2), (width, height // 2), (70, 70, 70), 1, cv2.LINE_AA)

    # Fixed threshold boundary lines (Red)
    mx_left = max(0, min(width - 1, int(width / 2 + threshold_left_x * 2500)))
    mx_right = max(0, min(width - 1, int(width / 2 + threshold_right_x * 2500)))
    my_top = max(0, min(height - 1, int(height / 2 - threshold_y * 2500)))
    my_bottom = max(0, min(height - 1, int(height / 2 + threshold_y * 2500)))

    cv2.line(gaze_map, (mx_left, 0), (mx_left, height), (0, 0, 180), 2, cv2.LINE_AA)
    cv2.line(gaze_map, (mx_right, 0), (mx_right, height), (0, 0, 180), 2, cv2.LINE_AA)
    cv2.line(gaze_map, (0, my_top), (width, my_top), (0, 0, 180), 1, cv2.LINE_AA)
    cv2.line(gaze_map, (0, my_bottom), (width, my_bottom), (0, 0, 180), 1, cv2.LINE_AA)

    # Screen Display bounds (Cyan)
    dev_left = max(0, min(width - 1, int(width / 2 + threshold_left_x * 1.1 * 2500)))
    dev_right = max(0, min(width - 1, int(width / 2 + threshold_right_x * 1.1 * 2500)))
    dev_top = max(0, min(height - 1, int(height / 2 - threshold_y * 1.1 * 2500)))
    dev_bottom = max(0, min(height - 1, int(height / 2 + threshold_y * 1.1 * 2500)))

    cv2.line(gaze_map, (dev_left, 0), (dev_left, height), (240, 200, 0), 1, cv2.LINE_AA)
    cv2.line(gaze_map, (dev_right, 0), (dev_right, height), (240, 200, 0), 1, cv2.LINE_AA)
    cv2.line(gaze_map, (0, dev_top), (width, dev_top), (240, 200, 0), 1, cv2.LINE_AA)
    cv2.line(gaze_map, (0, dev_bottom), (width, dev_bottom), (240, 200, 0), 1, cv2.LINE_AA)

    points = [(p[0], p[1]) for p in points_with_time]
    if len(points) > 0:
        num_pts = len(points)
        for i, (x, y) in enumerate(points):
            alpha = (i + 1) / num_pts
            color = (int(80 + 175 * alpha), int(80 + 175 * alpha), int(20 * alpha))
            radius = 3 if i < num_pts - 10 else 4
            cv2.circle(gaze_map, (x, y), radius, color, -1, cv2.LINE_AA)

        pts_arr = np.array(points)
        avg_x = int(np.mean(pts_arr[:, 0]))
        avg_y = int(np.mean(pts_arr[:, 1]))
        std_x = np.std(pts_arr[:, 0])
        std_y = np.std(pts_arr[:, 1])
        spread = math.hypot(std_x, std_y)

        # Average center marker
        cv2.line(gaze_map, (0, avg_y), (width, avg_y), (0, 140, 255), 1, cv2.LINE_AA)
        cv2.line(gaze_map, (avg_x, 0), (avg_x, height), (0, 140, 255), 1, cv2.LINE_AA)
        cv2.circle(gaze_map, (avg_x, avg_y), 8, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(gaze_map, (avg_x, avg_y), 14, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.putText(gaze_map, f"Gaze Points: {num_pts} | Spread: {spread:.1f}px", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(gaze_map, f"Average Gaze Center: ({avg_x}, {avg_y})", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(gaze_map, "Awaiting Candidate Gaze Telemetry...", (width // 2 - 150, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    cv2.putText(gaze_map, "GAZE MOTION DENSITY CHART", (width - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 240), 1, cv2.LINE_AA)

    if cheat_alarm:
        cv2.rectangle(gaze_map, (0, 0), (width, height), (0, 0, 255), 8)
        cv2.putText(gaze_map, "OFF-SCREEN LOOK ALERT", (width // 2 - 120, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    return gaze_map


# Button click state for calibration screen
button_clicked = False

def mouse_callback(event, x, y, flags, param):
    global button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if 200 <= x <= 440 and 240 <= y <= 290:
            button_clicked = True


def main():
    global button_clicked

    try:
        tracker = FaceMeshTracker()
    except Exception as e:
        print(f"Failed to initialize FaceMeshTracker: {e}")
        return

    try:
        object_detector = MediaPipeObjectDetector()
        print("MediaPipe Object Detector initialized.")
    except Exception as e:
        print(f"Object Detector initialization skipped: {e}")
        object_detector = None

    try:
        device_scanner = DeviceProctorScanner(audit_interval=600.0)
        print("Device Proctoring Scanner (BLE & Wi-Fi) active.")
    except Exception as e:
        print(f"Device Scanner initialization skipped: {e}")
        device_scanner = None

    # Camera acquisition
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        tracker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Windows setup - ONLY ONE focused window for the candidate
    cv2.namedWindow("Eye & Pupil Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye & Pupil Tracker", 640, 480)
    cv2.setMouseCallback("Eye & Pupil Tracker", mouse_callback)

    # State & Config
    state = 'WELCOME'
    config = load_config()
    evidence_mode = config.get("evidence_mode", "both")
    gaze_away_threshold = config.get("gaze_away_threshold_sec", 3.0)
    threshold_left_x = config.get("threshold_left_x", -0.12)
    threshold_right_x = config.get("threshold_right_x", 0.12)
    threshold_y = config.get("threshold_y", 0.09)
    candidate_name = config.get("candidate_name", "Student_01")

    # Violation Counters
    looked_left_count = 0
    looked_right_count = 0
    looked_up_count = 0
    looked_down_count = 0
    cell_phone_count = 0
    multiple_people_count = 0
    no_face_count = 0
    unauthorized_device_count = 0

    gaze_violation_active = False
    phone_violation_active = False
    multi_violation_active = False
    noface_violation_active = False
    dev_violation_active = False
    current_gaze_direction = "CENTER"
    no_user_consecutive = 0
    multi_person_consecutive = 0

    # Frequent Peeking Anomaly Engine (>7 glances in 60s)
    glance_events = deque()
    current_away_frames = 0
    peek_latched = False
    peek_direction = "CENTER"
    last_glance_time = 0.0
    repeated_peeking_count = 0
    peeking_alarm_active = False
    peeking_warning_start_time = 0.0

    # Calibration variables
    max_retries = 3
    center_retry_count = 0
    stage_start_time = 0.0
    retry_delay_start = 0.0
    retry_reason = ""
    calibration_offsets_x = []
    calibration_offsets_y = []
    bias_x = 0.0
    bias_y = 0.0

    # Sliding windows
    gaze_history = deque()
    map_w, map_h = 800, 600
    gaze_map_points = deque(maxlen=500)
    cheat_alarm = False
    away_time = 0.0

    frame_count = 0
    person_count = 1
    phone_detected = False
    detections = []
    obj_alarm_triggered = False
    obj_alarm_text = ""
    exam_recording_initialized = False
    last_stream_save_time = 0.0

    prev_time = time.time()
    fps = 0.0

    print("\n---------------- ProctorVision Candidate Exam Active ----------------")
    print(f"Candidate: {candidate_name} | Evidence Mode: {evidence_mode.upper()}")
    print("Press 'q' or 'ESC' to conclude exam.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1) # Mirror
        img_h, img_w, _ = frame.shape

        if state in ['WELCOME', 'CENTER_CALIB', 'RETRY_DELAY']:
            display_frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            display_frame[:] = (18, 18, 18)
        else:
            display_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks = tracker.process(rgb_frame)

        # Draw Clean Minimal Header
        cv2.rectangle(display_frame, (0, 0), (img_w, 45), (15, 15, 15), -1)
        cv2.line(display_frame, (0, 45), (img_w, 45), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "PROCTORVISION AI EXAM CLIENT", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 2, cv2.LINE_AA)
        status_tag = "LIVE MONITORING" if state == 'RUNNING' else "SETUP & CALIBRATION"
        cv2.putText(display_frame, status_tag, (img_w - 180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 150) if state == 'RUNNING' else (0, 200, 255), 1, cv2.LINE_AA)

        # Baseline BLE / Network Scan during setup
        if device_scanner and not device_scanner.get_status()['baseline_completed'] and not device_scanner.get_status()['is_scanning']:
            device_scanner.start_baseline_scan_async()

        # ---------------------------------------------------------
        # STATE 1: WELCOME & INSTRUCTIONS
        # ---------------------------------------------------------
        if state == 'WELCOME':
            cv2.putText(display_frame, "GAZE CALIBRATION SETUP", (img_w // 2 - 130, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 240), 2, cv2.LINE_AA)

            instructions = [
                "1. Keep your head upright and face the camera directly.",
                "2. Look directly at the green center calibration dot.",
                "3. Hold your gaze for 2 seconds to calibrate neutral bias.",
                "4. Your exam session will start automatically."
            ]
            for idx, text in enumerate(instructions):
                cv2.putText(display_frame, text, (40, 150 + idx * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Emerald Green Start Button
            cv2.rectangle(display_frame, (200, 240), (440, 290), (0, 180, 100), -1, cv2.LINE_AA)
            cv2.rectangle(display_frame, (200, 240), (440, 290), (0, 255, 150), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "START CALIBRATION", (225, 271),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            key = cv2.waitKey(1) & 0xFF
            if button_clicked or key == ord(' '):
                button_clicked = False
                state = 'CENTER_CALIB'
                stage_start_time = time.time()
                calibration_offsets_x.clear()
                calibration_offsets_y.clear()
                center_retry_count = 0

        # ---------------------------------------------------------
        # STATE 2: CENTER CALIBRATION
        # ---------------------------------------------------------
        elif state == 'CENTER_CALIB':
            cx, cy = img_w // 2, img_h // 2
            cv2.circle(display_frame, (cx, cy), 15, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)

            cv2.putText(display_frame, "LOOK DIRECTLY AT THE CENTER DOT", (img_w // 2 - 165, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)

            progress = min(1.0, len(calibration_offsets_x) / 30.0)
            cv2.putText(display_frame, f"Calibrating Center: {int(progress * 100)}%", (img_w // 2 - 100, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            bar_w = 200
            bar_x = img_w // 2 - bar_w // 2
            cv2.rectangle(display_frame, (bar_x, 380), (bar_x + bar_w, 390), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (bar_x, 380), (bar_x + int(bar_w * progress), 390), (0, 255, 0), -1)

            if face_landmarks:
                metrics = extract_eye_metrics(face_landmarks, img_w, img_h)
                if not metrics["blink"]:
                    calibration_offsets_x.append(metrics["raw_offset_x"])
                    calibration_offsets_y.append(metrics["raw_offset_y"])
            else:
                cv2.putText(display_frame, "Face not detected - Center face in camera view", (img_w // 2 - 160, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

            if len(calibration_offsets_x) >= 30:
                sorted_x = sorted(calibration_offsets_x)
                sorted_y = sorted(calibration_offsets_y)
                trim_k = max(1, len(sorted_x) // 10)
                clean_x = sorted_x[trim_k:-trim_k]
                clean_y = sorted_y[trim_k:-trim_k]

                std_x = np.std(clean_x)
                std_y = np.std(clean_y)

                if std_x > 0.08 or std_y > 0.08:
                    center_retry_count += 1
                    if center_retry_count < max_retries:
                        state = 'RETRY_DELAY'
                        retry_reason = "Excessive movement. Keep steady and look at the center dot."
                        retry_delay_start = time.time()
                    else:
                        bias_x = float(np.median(calibration_offsets_x))
                        bias_y = float(np.median(calibration_offsets_y))
                        state = 'RUNNING'
                else:
                    bias_x = float(np.median(clean_x))
                    bias_y = float(np.median(clean_y))
                    state = 'RUNNING'
                    print(f"[Calibration] Success! Calibrated Bias: X={bias_x:.4f}, Y={bias_y:.4f}")

            elif time.time() - stage_start_time > 12.0:
                if len(calibration_offsets_x) >= 15:
                    bias_x = float(np.median(calibration_offsets_x))
                    bias_y = float(np.median(calibration_offsets_y))
                else:
                    bias_x = 0.0
                    bias_y = 0.0
                state = 'RUNNING'

        # ---------------------------------------------------------
        # STATE 3: RETRY DELAY
        # ---------------------------------------------------------
        elif state == 'RETRY_DELAY':
            cv2.putText(display_frame, "CALIBRATION ATTEMPT FAILED", (img_w // 2 - 170, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, retry_reason, (img_w // 2 - len(retry_reason)*4, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            rem = max(0.0, 2.5 - (time.time() - retry_delay_start))
            cv2.putText(display_frame, f"Restarting in {rem:.1f}s...", (img_w // 2 - 70, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)

            if time.time() - retry_delay_start >= 2.5:
                calibration_offsets_x.clear()
                calibration_offsets_y.clear()
                state = 'CENTER_CALIB'
                stage_start_time = time.time()

        # ---------------------------------------------------------
        # STATE 4: RUNNING LIVE PROCTORING
        # ---------------------------------------------------------
        elif state == 'RUNNING':
            if not exam_recording_initialized:
                start_full_exam_recording(img_w, img_h, candidate_name=candidate_name)
                exam_recording_initialized = True

            blink = False
            calibrated_x = 0.0
            calibrated_y = 0.0
            gaze_state = "Center"
            current_gaze_direction = "CENTER"
            is_away = False

            if face_landmarks:
                metrics = extract_eye_metrics(face_landmarks, img_w, img_h)
                blink = metrics["blink"]

                # Draw subtle pupil circles and contours
                for pt in metrics["l_contour_pts"] + metrics["r_contour_pts"]:
                    cv2.circle(display_frame, pt, 1, (255, 255, 0), -1, cv2.LINE_AA)

                cv2.circle(display_frame, metrics["l_iris_center"], metrics["l_radius"], (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(display_frame, metrics["r_iris_center"], metrics["r_radius"], (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(display_frame, metrics["l_iris_center"], 2, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, metrics["r_iris_center"], 2, (0, 255, 0), -1, cv2.LINE_AA)

                # Normalized calibrated pupil offsets
                calibrated_x = metrics["raw_offset_x"] - bias_x
                calibrated_y = metrics["raw_offset_y"] - bias_y

                # Head Pose Estimation via facial anchor geometry
                head_left, head_right, head_up, head_down = False, False, False, False
                try:
                    nose_lm = face_landmarks[1]
                    l_cheek = face_landmarks[234]
                    r_cheek = face_landmarks[454]
                    forehead = face_landmarks[10]
                    chin = face_landmarks[152]

                    face_span_x = max(0.001, abs(r_cheek.x - l_cheek.x))
                    face_span_y = max(0.001, abs(chin.y - forehead.y))

                    nose_ratio_x = (nose_lm.x - l_cheek.x) / face_span_x
                    nose_ratio_y = (nose_lm.y - forehead.y) / face_span_y

                    if nose_ratio_x < 0.28: head_left = True
                    elif nose_ratio_x > 0.72: head_right = True

                    if nose_ratio_y > 0.76: head_down = True
                    elif nose_ratio_y < 0.28: head_up = True
                except Exception:
                    pass

                # Evaluate directional gaze state
                if calibrated_x > threshold_right_x or head_right:
                    gaze_state = "Looking Right"
                    current_gaze_direction = "RIGHT"
                    is_away = True
                elif calibrated_x < threshold_left_x or head_left:
                    gaze_state = "Looking Left"
                    current_gaze_direction = "LEFT"
                    is_away = True

                if calibrated_y > threshold_y or head_down:
                    gaze_state += " / Down" if gaze_state != "Center" else "Looking Down"
                    if current_gaze_direction == "CENTER":
                        current_gaze_direction = "DOWN"
                    is_away = True
                elif calibrated_y < -threshold_y or head_up:
                    gaze_state += " / Up" if gaze_state != "Center" else "Looking Up"
                    if current_gaze_direction == "CENTER":
                        current_gaze_direction = "UP"
                    is_away = True

                if not blink:
                    gaze_history.append((calibrated_x, calibrated_y, time.time(), is_away))
                    mx = max(0, min(map_w - 1, int(map_w / 2 + calibrated_x * 2500)))
                    my = max(0, min(map_h - 1, int(map_h / 2 + calibrated_y * 2500)))
                    gaze_map_points.append((mx, my, time.time()))

                state_text = "BLINKING" if blink else gaze_state.upper()
                state_color = (0, 0, 255) if blink else ((0, 255, 0) if gaze_state == "Center" else (255, 120, 0))
                cv2.putText(display_frame, f"GAZE: {state_text}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "NO FACE DETECTED", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

            # Stopwatch Look-Away Duration
            now = time.time()
            window_duration = max(8.0, gaze_away_threshold * 1.5)
            while gaze_history and now - gaze_history[0][2] > window_duration:
                gaze_history.popleft()
            while gaze_map_points and now - gaze_map_points[0][2] > window_duration:
                gaze_map_points.popleft()

            away_time = 0.0
            if len(gaze_history) > 0:
                for i in range(len(gaze_history) - 1):
                    if gaze_history[i][3]:
                        away_time += (gaze_history[i+1][2] - gaze_history[i][2])
                if gaze_history[-1][3]:
                    away_time += (now - gaze_history[-1][2])

            # Object Detection Proctoring (Every 15 frames)
            if object_detector and (frame_count % 15 == 0):
                try:
                    person_count, phone_detected, detections = object_detector.detect_objects(rgb_frame)
                    if person_count == 0: no_user_consecutive += 1
                    else: no_user_consecutive = 0

                    if person_count > 1: multi_person_consecutive += 1
                    else: multi_person_consecutive = 0

                    obj_alarm_triggered = False
                    obj_alarm_text = ""
                    if no_user_consecutive >= 2:
                        obj_alarm_triggered = True
                        obj_alarm_text = "ALARM: NO USER DETECTED"
                    elif multi_person_consecutive >= 2:
                        obj_alarm_triggered = True
                        obj_alarm_text = "ALARM: MULTIPLE PEOPLE DETECTED"
                    elif phone_detected:
                        obj_alarm_triggered = True
                        obj_alarm_text = "ALARM: MOBILE PHONE DETECTED"
                except Exception as e:
                    print(f"Object Detector error: {e}")

            # Draw detection bounding boxes if violations present
            if object_detector:
                for detection in detections:
                    category = detection.categories[0]
                    label = category.category_name.lower()
                    if label in ["person", "cell phone"]:
                        bbox = detection.bounding_box
                        bx, by, bw, bh = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(display_frame, label.upper(), (bx + 4, by + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            # Device Scanner Audit Check
            dev_alarm_triggered = False
            dev_alarm_text = ""
            if device_scanner:
                device_scanner.check_and_trigger_periodic_audit(now)
                dev_status = device_scanner.get_status()
                dev_alarm_triggered = dev_status['alarm_triggered']
                if dev_alarm_triggered:
                    dev_alarm_text = dev_status['alarm_message']

            # Helper for incident gaze snapshot
            def make_incident_density():
                recent_pts = [p for p in gaze_map_points if (now - p[2] <= 10.0)]
                return draw_density_map(recent_pts, 600, 450, cheat_alarm=True,
                                        threshold_left_x=threshold_left_x, threshold_right_x=threshold_right_x,
                                        threshold_y=threshold_y)

            # 1. Peeking Anomaly Engine (>7 peeks in 60s)
            if is_away:
                current_away_frames += 1
                if current_away_frames >= 8 and not peek_latched:
                    peek_latched = True
                    peek_direction = current_gaze_direction
            else:
                if peek_latched and (now - last_glance_time > 0.8):
                    glance_events.append((now, peek_direction))
                    last_glance_time = now
                peek_latched = False
                current_away_frames = 0

            while glance_events and (now - glance_events[0][0] > 60.0):
                glance_events.popleft()

            if len(glance_events) > 7:
                if not peeking_alarm_active:
                    peeking_alarm_active = True
                    peeking_warning_start_time = now
                    repeated_peeking_count += 1
                    dirs = [g[1] for g in glance_events]
                    peek_dir = max(set(dirs), key=dirs.count) if dirs else "OFF-SCREEN"
                    record_violation_event(
                        event_type="REPEATED_PEEKING_ANOMALY",
                        direction=peek_dir,
                        message=f"Frequent peeking anomaly: Candidate glanced off-screen {len(glance_events)} times in 1m ({peek_dir})",
                        frame=display_frame,
                        frame_width=img_w,
                        frame_height=img_h,
                        clip_density_img=make_incident_density()
                    )
            elif len(glance_events) < 4 and (now - peeking_warning_start_time > 6.0):
                peeking_alarm_active = False

            # 2. Sustained Gaze Violation
            gaze_alarm = (away_time >= gaze_away_threshold)
            if gaze_alarm:
                if not gaze_violation_active:
                    gaze_violation_active = True
                    if current_gaze_direction == "LEFT": looked_left_count += 1
                    elif current_gaze_direction == "RIGHT": looked_right_count += 1
                    elif current_gaze_direction == "UP": looked_up_count += 1
                    elif current_gaze_direction == "DOWN": looked_down_count += 1
                    else: looked_left_count += 1; current_gaze_direction = "LEFT"

                    record_violation_event(
                        event_type=f"GAZE_LOOK_{current_gaze_direction}",
                        direction=current_gaze_direction,
                        message=f"Candidate looked off-screen ({current_gaze_direction}) for {away_time:.1f}s",
                        frame=display_frame,
                        frame_width=img_w,
                        frame_height=img_h,
                        clip_density_img=make_incident_density()
                    )
            elif away_time < 0.35:
                gaze_violation_active = False

            # 3. Object Detection Violation Dispatcher
            if obj_alarm_triggered:
                if person_count == 0 and not noface_violation_active:
                    noface_violation_active = True
                    no_face_count += 1
                    record_violation_event("NO_USER_DETECTED", "NONE", "Candidate left camera field of view",
                                           display_frame, img_w, img_h, make_incident_density())
                elif person_count > 1 and not multi_violation_active:
                    multi_violation_active = True
                    multiple_people_count += 1
                    record_violation_event("MULTIPLE_PEOPLE", "NONE", f"Multiple people ({person_count}) detected",
                                           display_frame, img_w, img_h, make_incident_density())
                elif phone_detected and not phone_violation_active:
                    phone_violation_active = True
                    cell_phone_count += 1
                    record_violation_event("CELL_PHONE_DETECTED", "NONE", "Mobile phone detected in frame",
                                           display_frame, img_w, img_h, make_incident_density())
            else:
                noface_violation_active = False
                multi_violation_active = False
                phone_violation_active = False

            # 4. Device Scanner Violation
            if dev_alarm_triggered and not dev_violation_active:
                dev_violation_active = True
                unauthorized_device_count += 1
                record_violation_event("UNAUTHORIZED_DEVICE", "NONE", dev_alarm_text,
                                       display_frame, img_w, img_h, make_incident_density())
            elif not dev_alarm_triggered:
                dev_violation_active = False

            cheat_alarm = gaze_alarm or obj_alarm_triggered or dev_alarm_triggered

            # Visual Feedback on Candidate UI
            if cheat_alarm:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (img_w, img_h), (0, 0, 100), -1)
                cv2.addWeighted(overlay, 0.22, display_frame, 0.78, 0, display_frame)

                alert_title = obj_alarm_text if obj_alarm_triggered else (dev_alarm_text if dev_alarm_triggered else f"LOOKING {current_gaze_direction}")
                cv2.rectangle(display_frame, (img_w // 2 - 200, img_h - 60), (img_w // 2 + 200, img_h - 15), (0, 0, 200), -1, cv2.LINE_AA)
                cv2.putText(display_frame, alert_title, (img_w // 2 - len(alert_title)*5, img_h - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            elif away_time > 0.5:
                cv2.putText(display_frame, f"Gaze Alert ({current_gaze_direction}): {away_time:.1f}s / {gaze_away_threshold:.1f}s",
                            (img_w // 2 - 110, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

            # Generate Live Density Map in background for Admin Streaming
            if now - last_stream_save_time >= 0.15:
                last_stream_save_time = now
                live_gaze_chart = draw_density_map(gaze_map_points, map_w, map_h, cheat_alarm=cheat_alarm,
                                                   threshold_left_x=threshold_left_x, threshold_right_x=threshold_right_x,
                                                   threshold_y=threshold_y)
                save_live_frames(display_frame, live_gaze_chart)

                total_viols = (looked_left_count + looked_right_count + looked_up_count + looked_down_count +
                               cell_phone_count + multiple_people_count + no_face_count + unauthorized_device_count +
                               repeated_peeking_count)
                update_live_telemetry({
                    "is_live": True,
                    "candidate_name": candidate_name,
                    "gaze_state": gaze_state,
                    "direction": current_gaze_direction,
                    "away_time": round(away_time, 1),
                    "gaze_threshold": gaze_away_threshold,
                    "fps": round(fps, 1),
                    "total_violations": total_viols,
                    "looked_left": looked_left_count,
                    "looked_right": looked_right_count,
                    "cell_phone": cell_phone_count,
                    "repeated_peeking": repeated_peeking_count,
                    "timestamp": datetime.datetime.now().isoformat()
                })

        # FPS Tracking
        curr_time = time.time()
        time_diff = curr_time - prev_time
        prev_time = curr_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0

        if exam_recording_initialized:
            process_frame(display_frame)

        # Show single candidate window
        cv2.imshow("Eye & Pupil Tracker", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            if cheat_alarm and evidence_mode in ["video", "both"]:
                trigger_cheat_recording(img_w, img_h, is_test_ending=True)
            break
        elif key == ord('c'):
            state = 'WELCOME'
            button_clicked = False
            gaze_map_points.clear()
            gaze_history.clear()
            calibration_offsets_x.clear()
            calibration_offsets_y.clear()
            center_retry_count = 0
            cheat_alarm = False
        elif key == ord('r'):
            gaze_map_points.clear()
            gaze_history.clear()
            cheat_alarm = False

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    # Finalize Session & Density Map
    if exam_recording_initialized:
        print("[System] Finalizing exam session & exporting metadata...")
        summary_counts = {
            "total_violations": (looked_left_count + looked_right_count + looked_up_count + looked_down_count +
                                 cell_phone_count + multiple_people_count + no_face_count + unauthorized_device_count +
                                 repeated_peeking_count),
            "looked_left": looked_left_count,
            "looked_right": looked_right_count,
            "looked_up": looked_up_count,
            "looked_down": looked_down_count,
            "repeated_peeking": repeated_peeking_count,
            "cell_phone": cell_phone_count,
            "multiple_people": multiple_people_count,
            "no_face": no_face_count,
            "unauthorized_device": unauthorized_device_count
        }
        final_density_map = draw_density_map(gaze_map_points, map_w, map_h, cheat_alarm=False,
                                             threshold_left_x=threshold_left_x, threshold_right_x=threshold_right_x,
                                             threshold_y=threshold_y)
        compressed = run_end_of_exam_compression(summary_counts=summary_counts, density_map_img=final_density_map)
        print(f"[System] Session compression complete. Output: {compressed}")

    print("ProctorVision candidate client exited successfully.")


if __name__ == "__main__":
    main()
