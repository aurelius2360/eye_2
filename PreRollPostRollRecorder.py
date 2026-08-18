import os
import sys
import time
import json
import shutil
import subprocess
from collections import deque, Counter
import datetime
import cv2

from model_utils import get_resource_path

CONFIG_FILE = "proctor_config.json"
ACTIVE_SESSION_FILE = os.path.join("sessions", "active_session.json")

def load_config():
    """Loads configuration from proctor_config.json with safe defaults."""
    default_cfg = {
        "evidence_mode": "both",  # "video", "photo", "both"
        "capture_full_exam_video": True,
        "gaze_away_threshold_sec": 3.0,
        "threshold_left_x": -0.12,
        "threshold_right_x": 0.12,
        "threshold_y": 0.09,
        "device_audit_interval_sec": 600.0,
        "candidate_name": "Student_01"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                default_cfg.update(json.load(f))
        except Exception as e:
            print(f"[Config] Error reading {CONFIG_FILE}: {e}")
    return default_cfg

def save_config(cfg):
    """Saves configuration to proctor_config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print(f"[Config] Error saving {CONFIG_FILE}: {e}")
        return False


def find_local_ffmpeg():
    """Locates the FFmpeg binary via PATH, bundled path, or imageio_ffmpeg."""
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path

    for name in ["ffmpeg.exe", "ffmpeg"]:
        bundled = get_resource_path(name)
        if os.path.isfile(bundled):
            return bundled

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    return "ffmpeg"


# --- GLOBAL STATE & BUFFERING ---
history_buffer = deque(maxlen=300) # 10s pre-roll at 30 FPS

is_recording_cheat = False
cheat_record_frames_left = 0
cheat_events_to_process = []
video_writer = None
current_cheat_file = ""

full_exam_writer = None
full_exam_file = ""
exam_start_time = 0.0

exam_base_name = ""
anomaly_count = 0
photo_count = 0

session_dir = ""
session_events = []
current_config = load_config()


def start_full_exam_recording(frame_width, frame_height, candidate_name=None):
    """
    Initializes continuous full-exam recording and sets up self-contained session directory.
    """
    global full_exam_writer, full_exam_file, exam_start_time, exam_base_name, anomaly_count, photo_count
    global session_dir, session_events, current_config

    current_config = load_config()
    if candidate_name is None:
        candidate_name = current_config.get("candidate_name", "Student_01")

    exam_start_time = time.time()
    exam_base_name = f"exam_{int(exam_start_time)}"
    anomaly_count = 0
    photo_count = 0
    session_events = []

    # Ensure self-contained directory tree
    session_dir = os.path.join("sessions", exam_base_name)
    os.makedirs(os.path.join(session_dir, "photos"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "videos"), exist_ok=True)

    # Register active live session
    active_meta = {
        "session_id": exam_base_name,
        "candidate_name": candidate_name,
        "start_time": datetime.datetime.fromtimestamp(exam_start_time).isoformat(),
        "is_live": True,
        "session_dir": session_dir,
        "evidence_mode": current_config.get("evidence_mode", "both")
    }
    try:
        with open(ACTIVE_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(active_meta, f, indent=2)
    except Exception as e:
        print(f"[System] Error writing active session: {e}")

    if current_config.get("capture_full_exam_video", True):
        full_exam_file = os.path.join(session_dir, f"{exam_base_name}_full_raw.mp4")
        print(f"[System] Starting Full Exam continuous recording: {full_exam_file}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        full_exam_writer = cv2.VideoWriter(full_exam_file, fourcc, 30.0, (frame_width, frame_height))
    else:
        full_exam_file = ""
        full_exam_writer = None


def update_live_telemetry(telemetry_data):
    """
    Writes live candidate telemetry and status to the session directory for the Admin Portal.
    """
    global session_dir, exam_base_name
    if not session_dir:
        return
    
    telemetry_file = os.path.join(session_dir, "live_telemetry.json")
    try:
        with open(telemetry_file, "w", encoding="utf-8") as f:
            json.dump(telemetry_data, f)
    except Exception:
        pass


def save_live_frames(candidate_frame, gaze_density_frame):
    """
    Saves the latest live frame and gaze density chart for real-time Admin streaming.
    """
    global session_dir
    if not session_dir:
        return
    try:
        if candidate_frame is not None:
            cv2.imwrite(os.path.join(session_dir, "live_feed.jpg"), candidate_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if gaze_density_frame is not None:
            cv2.imwrite(os.path.join(session_dir, "live_gaze.jpg"), gaze_density_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        pass


def capture_violation_snapshot(frame, violation_title, details=""):
    """
    Captures a high-resolution snapshot with watermark directly into the session photos directory.
    """
    global photo_count, exam_base_name, session_dir
    photo_count += 1

    snap_frame = frame.copy()
    h, w, _ = snap_frame.shape

    # Render evidence metadata banner
    cv2.rectangle(snap_frame, (0, h - 35), (w, h), (10, 10, 10), -1)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp_text = f"EVIDENCE #{photo_count} | {violation_title.upper()} | {timestamp_str}"
    cv2.putText(snap_frame, stamp_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 240), 1, cv2.LINE_AA)

    rel_filename = f"{exam_base_name}_photo_{photo_count}.jpg"
    session_photo_path = os.path.join(session_dir, "photos", rel_filename) if session_dir else rel_filename
    cv2.imwrite(session_photo_path, snap_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[System] Violation Snapshot #{photo_count} saved: {session_photo_path}")
    return session_photo_path


def trigger_cheat_recording(frame_width, frame_height, is_test_ending=False):
    """
    Triggers 10s pre-roll + 10s post-roll recording when an anomaly occurs.
    """
    global is_recording_cheat, cheat_record_frames_left, video_writer, current_cheat_file
    global exam_start_time, exam_base_name, anomaly_count, session_dir

    if is_recording_cheat:
        return ""

    anomaly_count += 1
    video_dir = os.path.join(session_dir, "videos") if session_dir else "videos"
    os.makedirs(video_dir, exist_ok=True)
    current_cheat_file = os.path.join(video_dir, f"{exam_base_name}_{anomaly_count}_raw.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(current_cheat_file, fourcc, 30.0, (frame_width, frame_height))

    elapsed_time = time.time() - exam_start_time if exam_start_time > 0.0 else 0.0

    if is_test_ending:
        print(f"[System] End-of-Test Anomaly! Saving pre-roll history to {current_cheat_file}")
        for frame in list(history_buffer):
            video_writer.write(frame)
        video_writer.release()
        video_writer = None
        cheat_events_to_process.append(current_cheat_file)
    else:
        # Write pre-roll buffer
        for frame in list(history_buffer):
            video_writer.write(frame)
        is_recording_cheat = True
        cheat_record_frames_left = 300

    return current_cheat_file


def record_violation_event(event_type, direction, message, frame, frame_width, frame_height, clip_density_img=None):
    """
    Central violation event dispatcher saving photo, video clip, and incident gaze density chart.
    """
    global session_events, current_config, exam_start_time, anomaly_count, exam_base_name, session_dir

    mode = current_config.get("evidence_mode", "both").lower()
    photo_file = ""
    video_raw_file = ""
    clip_density_path = ""

    # 1. Snapshot
    if mode in ["photo", "both"] and frame is not None:
        title = f"{event_type} ({direction})" if direction != "NONE" else event_type
        photo_file = capture_violation_snapshot(frame, title, message)

    # 2. Incident Gaze Density Plot
    if clip_density_img is not None and session_dir:
        try:
            ev_id = len(session_events) + 1
            density_filename = f"{exam_base_name}_clip_{ev_id}_density.png"
            clip_density_path = os.path.join(session_dir, "photos", density_filename)
            cv2.imwrite(clip_density_path, clip_density_img)
        except Exception as e:
            print(f"[System] Error saving clip density map: {e}")

    # 3. Video
    if mode in ["video", "both"]:
        video_raw_file = trigger_cheat_recording(frame_width, frame_height)

    elapsed = time.time() - exam_start_time if exam_start_time > 0 else 0.0
    expected_webm = video_raw_file.replace("_raw.mp4", ".webm") if video_raw_file else ""

    event_entry = {
        "id": len(session_events) + 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "type": event_type,
        "direction": direction,
        "message": message,
        "evidence_mode": mode,
        "photo_file": photo_file,
        "video_file": expected_webm,
        "clip_density_map": clip_density_path
    }
    session_events.append(event_entry)
    return event_entry


def process_frame(frame):
    """
    Buffers frames and writes to active full exam and anomaly recorders.
    """
    global is_recording_cheat, cheat_record_frames_left, video_writer, cheat_events_to_process, full_exam_writer

    history_buffer.append(frame)

    if full_exam_writer is not None and full_exam_writer.isOpened():
        full_exam_writer.write(frame)

    if is_recording_cheat and video_writer is not None and video_writer.isOpened():
        video_writer.write(frame)
        cheat_record_frames_left -= 1

        if cheat_record_frames_left == 0:
            video_writer.release()
            video_writer = None
            is_recording_cheat = False
            cheat_events_to_process.append(current_cheat_file)
            print(f"[System] Anomaly Video Clip complete: {current_cheat_file}")


def run_end_of_exam_compression(summary_counts=None, density_map_img=None):
    """
    Closes video streams, compresses clips via FFmpeg into WebM, saves density map,
    and writes session_meta.json.
    """
    global cheat_events_to_process, full_exam_writer, full_exam_file, video_writer
    global session_dir, session_events, current_config, exam_start_time, exam_base_name

    compressed_files = []

    if full_exam_writer is not None:
        full_exam_writer.release()
        full_exam_writer = None

    if video_writer is not None:
        video_writer.release()
        video_writer = None
        if current_cheat_file and current_cheat_file not in cheat_events_to_process:
            cheat_events_to_process.append(current_cheat_file)

    raw_files_to_compress = []
    final_full_video_path = ""
    if full_exam_file and os.path.exists(full_exam_file):
        dest_full = full_exam_file.replace("_raw.mp4", ".webm")
        raw_files_to_compress.append((full_exam_file, dest_full))
        final_full_video_path = dest_full

    for raw_file in cheat_events_to_process:
        if raw_file and os.path.exists(raw_file):
            dest_anomaly = raw_file.replace("_raw.mp4", ".webm")
            raw_files_to_compress.append((raw_file, dest_anomaly))

    ffmpeg_cmd = find_local_ffmpeg()
    print(f"[System] Compressing video evidence using FFmpeg: '{ffmpeg_cmd}'")

    for raw_file, webm_file in raw_files_to_compress:
        try:
            cmd = [
                ffmpeg_cmd, "-y",
                "-i", raw_file,
                "-vf", "scale=-1:720",
                "-c:v", "libvpx-vp9",
                "-crf", "35",
                "-r", "15",
                "-b:v", "4M",
                webm_file
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, creationflags=flags)

            if os.path.exists(webm_file) and os.path.getsize(webm_file) > 0:
                os.remove(raw_file)
                compressed_files.append(webm_file)
        except Exception as e:
            print(f"[System] Video compression error for '{raw_file}': {e}")

    # Save overall exam density map
    density_map_path = ""
    if density_map_img is not None and session_dir:
        density_map_path = os.path.join(session_dir, "density_map.png")
        cv2.imwrite(density_map_path, density_map_img)

    end_time = time.time()
    duration = end_time - exam_start_time if exam_start_time > 0 else 0.0

    if summary_counts is None:
        direction_counts = Counter(e.get("direction") for e in session_events)
        type_counts = Counter(e.get("type", "") for e in session_events)
        summary_counts = {
            "total_violations": len(session_events),
            "looked_left": direction_counts.get("LEFT", 0),
            "looked_right": direction_counts.get("RIGHT", 0),
            "looked_up": direction_counts.get("UP", 0),
            "looked_down": direction_counts.get("DOWN", 0),
            "repeated_peeking": type_counts.get("REPEATED_PEEKING_ANOMALY", 0),
            "cell_phone": sum(1 for e in session_events if "PHONE" in e.get("type", "")),
            "multiple_people": sum(1 for e in session_events if "MULTIPLE" in e.get("type", "")),
            "no_face": sum(1 for e in session_events if "NO_USER" in e.get("type", "") or "NO_FACE" in e.get("type", "")),
            "unauthorized_device": sum(1 for e in session_events if "DEVICE" in e.get("type", ""))
        }

    meta_record = {
        "session_id": exam_base_name,
        "candidate_name": current_config.get("candidate_name", "Student_01"),
        "start_time": datetime.datetime.fromtimestamp(exam_start_time).isoformat() if exam_start_time > 0 else datetime.datetime.now().isoformat(),
        "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": round(duration, 1),
        "evidence_mode": current_config.get("evidence_mode", "both"),
        "summary_counts": summary_counts,
        "events": session_events,
        "full_video": final_full_video_path,
        "density_map": density_map_path
    }

    if session_dir:
        meta_file = os.path.join(session_dir, "session_meta.json")
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_record, f, indent=2)
            print(f"[System] Session summary saved: {meta_file}")
        except Exception as e:
            print(f"[System] Error writing session metadata: {e}")

    # Mark active session as finished
    try:
        if os.path.exists(ACTIVE_SESSION_FILE):
            with open(ACTIVE_SESSION_FILE, "r", encoding="utf-8") as f:
                act = json.load(f)
            act["is_live"] = False
            with open(ACTIVE_SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(act, f, indent=2)
    except Exception:
        pass

    cheat_events_to_process = []
    full_exam_file = ""
    return compressed_files
