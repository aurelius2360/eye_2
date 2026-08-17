import os
import sys
import time
import collections
import cv2
import ffmpeg
import numpy as np

# 1. GLOBAL STATE & BUFFERING
# Rolling buffer for pre-roll (300 frames representing 10 seconds of history at 30 FPS)
history_buffer = collections.deque(maxlen=300)

# Global variables tracking cheat recording status
is_recording_cheat = False
cheat_record_frames_left = 0
cheat_events_to_process = []
video_writer = None
current_cheat_file = ""

# Full exam recording (Type 1) variables
full_exam_writer = None
full_exam_file = ""
exam_start_time = 0.0

# Base name and counter for folder and file organization
exam_base_name = ""
anomaly_count = 0


def find_local_ffmpeg():
    """
    Dynamically searches the current working directory, subdirectories,
    and PyInstaller temporary folder for a local FFmpeg executable.
    Falls back to imageio_ffmpeg wrapper if installed.
    """
    # 1. If running as a bundled PyInstaller executable, check temp directory first
    if hasattr(sys, '_MEIPASS'):
        for name in ["ffmpeg.exe", "ffmpeg"]:
            path = os.path.join(sys._MEIPASS, name)
            if os.path.exists(path) and os.path.isfile(path):
                return path

    cwd = os.path.abspath(".")
    # 2. Check current directory root
    for name in ["ffmpeg.exe", "ffmpeg"]:
        path = os.path.join(cwd, name)
        if os.path.exists(path) and os.path.isfile(path):
            return path
            
    # 3. Search subdirectories recursively
    for root, dirs, files in os.walk(cwd):
        # Skip virtualenv and git directories
        if any(ignored in root.lower() for ignored in ["jk", "venv", ".git", "__pycache__"]):
            continue
        for name in ["ffmpeg.exe", "ffmpeg"]:
            if name in files:
                return os.path.join(root, name)
                
    # 4. Fallback to imageio_ffmpeg pre-compiled binary wrapper if available
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(path):
            return path
    except ImportError:
        pass
        
    return None


# 2. LOGIC FOR THE LIVE CAMERA LOOP & TYPE INITIALIZATION
def start_full_exam_recording(frame_width, frame_height):
    """
    Initializes the full-exam continuous recording (Type 1).
    Saves raw files into the 'full_video' folder.
    """
    global full_exam_writer, full_exam_file, exam_start_time, exam_base_name, anomaly_count
    
    # Ensure folders exist
    os.makedirs("full_video", exist_ok=True)
    os.makedirs("iterations", exist_ok=True)
    
    exam_start_time = time.time()
    exam_base_name = f"exam_{int(exam_start_time)}"
    anomaly_count = 0
    
    full_exam_file = os.path.join("full_video", f"{exam_base_name}_raw.mp4")
    print(f"[System] Starting continuous Type 1 Full Exam recording: {full_exam_file}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    full_exam_writer = cv2.VideoWriter(full_exam_file, fourcc, 30.0, (frame_width, frame_height))


def trigger_cheat_recording(frame_width, frame_height, is_test_ending=False):
    """
    Activates when a proctoring anomaly is triggered (Type 2).
    Saves anomaly raw files into the 'iterations' folder using suffixes _1, _2, _3.
    """
    global is_recording_cheat, cheat_record_frames_left, video_writer, current_cheat_file, exam_start_time, exam_base_name, anomaly_count
    
    if is_recording_cheat:
        print("[System] Warning: Anomaly recording already in progress. Trigger ignored.")
        return
        
    # Ensure folders exist
    os.makedirs("iterations", exist_ok=True)
    
    anomaly_count += 1
    # Use the same base name as the full video but suffix with 1, 2, 3 for anomalies
    current_cheat_file = os.path.join("iterations", f"{exam_base_name}_{anomaly_count}_raw.mp4")
    
    # Initialize OpenCV VideoWriter targeting the raw .mp4 anomaly file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(current_cheat_file, fourcc, 30.0, (frame_width, frame_height))
    
    elapsed_time = time.time() - exam_start_time if exam_start_time > 0.0 else 0.0
    
    if is_test_ending:
        # Rule: End of test -> capture the BEFORE 10 seconds only (pre-roll dump and immediate finalize)
        print(f"[System] End-of-Test Anomaly! Saving BEFORE 10 seconds (pre-roll) to {current_cheat_file}")
        for frame in list(history_buffer):
            video_writer.write(frame)
        video_writer.release()
        video_writer = None
        cheat_events_to_process.append(current_cheat_file)
        
    elif elapsed_time < 10.0:
        # Rule: Start of test -> capture the NEXT 10 seconds only (post-roll live write, bypass pre-roll history)
        print(f"[System] Start-of-Test Anomaly (Elapsed: {elapsed_time:.1f}s)! Saving NEXT 10 seconds (post-roll) to {current_cheat_file}")
        is_recording_cheat = True
        cheat_record_frames_left = 300
        
    else:
        # Normal Case: capture BOTH BEFORE 10 seconds (pre-roll) and NEXT 10 seconds (post-roll)
        print(f"[System] Normal Anomaly! Saving BEFORE 10 seconds + NEXT 10 seconds to {current_cheat_file}")
        for frame in list(history_buffer):
            video_writer.write(frame)
        is_recording_cheat = True
        cheat_record_frames_left = 300


def process_frame(frame):
    """
    Processes a single frame inside the live monitoring loop.
    - Appends the frame to history.
    - Writes to the continuous full exam recording (Type 1) inside 'full_video'.
    - Writes to active anomaly VideoWriter (Type 2) inside 'iterations' if active.
    """
    global is_recording_cheat, cheat_record_frames_left, video_writer, cheat_events_to_process, full_exam_writer
    
    # 1. Continuously append frame to rolling pre-roll buffer
    history_buffer.append(frame)
    
    # 2. Write to full exam video (Type 1)
    if full_exam_writer is not None and full_exam_writer.isOpened():
        full_exam_writer.write(frame)
        
    # 3. Write to anomaly video (Type 2) if currently recording
    if is_recording_cheat:
        if video_writer is not None and video_writer.isOpened():
            video_writer.write(frame)
            cheat_record_frames_left -= 1
            
            # Post-roll complete
            if cheat_record_frames_left == 0:
                video_writer.release()
                video_writer = None
                is_recording_cheat = False
                cheat_events_to_process.append(current_cheat_file)
                print(f"[System] Post-roll complete. Raw anomaly recording saved: {current_cheat_file}")


# 3. LOGIC FOR END-OF-EXAM HOOK & COMPRESSION
def run_end_of_exam_compression():
    """
    Executes post-exam. Releases active writers, compresses raw .mp4 cache videos
    and full exam raw recording into highly optimized VP9/WebM format inside
    their respective folders (full_video and iterations), and purges raw cache files.
    """
    global cheat_events_to_process, full_exam_writer, full_exam_file, video_writer
    compressed_files = []
    
    # Release any active writers
    if full_exam_writer is not None:
        full_exam_writer.release()
        full_exam_writer = None
        
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        if current_cheat_file and current_cheat_file not in cheat_events_to_process:
            cheat_events_to_process.append(current_cheat_file)
            
    # Gather all raw files to process with their correct destination paths
    raw_files_to_compress = []
    if full_exam_file and os.path.exists(full_exam_file):
        # Compress from full_video/exam_<timestamp>_raw.mp4 to full_video/exam_<timestamp>.webm
        dest_full = full_exam_file.replace("_raw.mp4", ".webm")
        raw_files_to_compress.append((full_exam_file, dest_full))
        
    for raw_file in cheat_events_to_process:
        if raw_file and os.path.exists(raw_file):
            # Compress from iterations/exam_<timestamp>_<num>_raw.mp4 to iterations/exam_<timestamp>_<num>.webm
            dest_anomaly = raw_file.replace("_raw.mp4", ".webm")
            raw_files_to_compress.append((raw_file, dest_anomaly))
            
    # Dynamically locate the local FFmpeg binary or fallback to system path
    ffmpeg_cmd = find_local_ffmpeg() or 'ffmpeg'
    print(f"[System] Utilizing FFmpeg binary/path: '{ffmpeg_cmd}'")
    
    for raw_file, webm_file in raw_files_to_compress:
        print(f"[System] Compressing raw file '{raw_file}' to optimized '{webm_file}'...")
        
        try:
            # Construct the compression process graph using ffmpeg-python
            stream = ffmpeg.input(raw_file)
            
            # Downscale resolution to 720p maximum (scale=-1:720)
            stream = ffmpeg.filter(stream, 'scale', -1, 720)
            
            # Output options: VP9, CRF 35, 15 FPS, target video bitrate 4M
            stream = ffmpeg.output(
                stream,
                webm_file,
                vcodec='libvpx-vp9',
                crf=35,
                r=15,
                **{'b:v': '4M'}
            )
            
            # Run the process
            ffmpeg.run(stream, cmd=ffmpeg_cmd, overwrite_output=True, quiet=True)
            
            # If compression succeeded, delete original raw file
            if os.path.exists(webm_file) and os.path.getsize(webm_file) > 0:
                print(f"[System] Compression complete. Deleting raw file '{raw_file}' to save space...")
                os.remove(raw_file)
                compressed_files.append(webm_file)
            else:
                print(f"[System] Error: WebM output is empty or invalid for '{raw_file}'")
                
        except Exception as e:
            print(f"[System] Error compressing raw video '{raw_file}': {e}")
            
    # Clear the processing states
    cheat_events_to_process = []
    full_exam_file = ""
    return compressed_files


# --- SIMULATION BLOCK ---
def run_simulation():
    """
    Demonstrates Type 1 and Type 2 recordings including:
      - Continuous full exam recording inside 'full_video'
      - Start-of-test anomaly trigger (next 10 seconds post-roll only) inside 'iterations'
      - Normal anomaly trigger (both pre-roll and post-roll) inside 'iterations'
      - End-of-test anomaly trigger (before 10 seconds pre-roll only) inside 'iterations'
    """
    print("\n================ STARTING MONITORING SIMULATION ================")
    width, height = 640, 480
    
    # Initialize continuous full-exam recording
    start_full_exam_recording(width, height)
    
    # Scenario 1: Start-of-test anomaly (triggered in the first 5 seconds / 150 frames)
    print("\n--- Scenario 1: Start-of-Test Anomaly (First 5 seconds) ---")
    for frame_idx in range(120):
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, f"Frame {frame_idx:03d} (Start)", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        process_frame(dummy_frame)
        
    print("[Simulation] Anomaly triggered! (Only aftermath/next 10 seconds should be recorded)")
    trigger_cheat_recording(width, height)
    
    # Process 320 aftermath frames to satisfy post-roll and return to normal monitoring
    for frame_idx in range(120, 440):
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, f"Frame {frame_idx:03d} (Post-roll 1)", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        process_frame(dummy_frame)

    # Scenario 2: Normal anomaly (triggered after 10+ seconds / 300+ frames of activity)
    print("\n--- Scenario 2: Normal Anomaly (Middle of Exam) ---")
    # Feed more frames to build history
    for frame_idx in range(440, 800):
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, f"Frame {frame_idx:03d} (Normal)", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        process_frame(dummy_frame)
        
    print("[Simulation] Anomaly triggered! (Both pre-roll and post-roll should be recorded)")
    trigger_cheat_recording(width, height)
    
    # Process aftermath
    for frame_idx in range(800, 1120):
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, f"Frame {frame_idx:03d} (Post-roll 2)", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        process_frame(dummy_frame)

    # Scenario 3: End-of-test anomaly (triggered right before the test completes)
    print("\n--- Scenario 3: End-of-Test Anomaly (Exam Teardown) ---")
    for frame_idx in range(1120, 1450):
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, f"Frame {frame_idx:03d} (Ending)", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        process_frame(dummy_frame)
        
    print("[Simulation] Anomaly triggered at end of test! (Only before 10 seconds pre-roll should be saved)")
    trigger_cheat_recording(width, height, is_test_ending=True)
    
    print(f"\n[Simulation] Raw cache recordings ready for processing: {cheat_events_to_process}")
    
    # Compress all recordings
    print("\n[Simulation] Running final end-of-exam compression hook...")
    webm_list = run_end_of_exam_compression()
    print(f"[Simulation] Final WebM files ready for upload: {webm_list}")
    print("================================================================\n")


if __name__ == "__main__":
    run_simulation()
