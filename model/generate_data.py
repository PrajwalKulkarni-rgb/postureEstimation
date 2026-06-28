import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import glob
import os
import logging
from utils.normalization import normalize_skeleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_ROOT = "Data/" 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_X = os.path.join(SCRIPT_DIR, "trainable_data/x_train.npy")
OUTPUT_Y = os.path.join(SCRIPT_DIR, "trainable_data/y_train.npy")
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pose_landmarker_full.task'))
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5)
detector = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    """Calculates 3D angle at vertex b."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def get_bio_mechanical_label(lms):
    """
    Determines risk based on EAWS (European Assembly Worksheet) and NIOSH standards.
    
    EAWS Section 4 (Postures):
    - Neutral: 0-20 deg flexion (Angle 160-180)
    - Bent (Class 3): 20-60 deg flexion (Angle 120-160)
    - Strongly Bent (Class 4): >60 deg flexion (Angle < 120)
    """
    def get_midpoint(idx1, idx2):
        return [(lms[idx1].x + lms[idx2].x) / 2,
                (lms[idx1].y + lms[idx2].y) / 2,
                (lms[idx1].z + lms[idx2].z) / 2]

    # Extract 3D Midpoints for a bias-free skeletal center
    shoulder = get_midpoint(11, 12)
    hip = get_midpoint(23, 24)
    knee = get_midpoint(25, 26)
    ankle = get_midpoint(27, 28)
    
    # 1. Calculate EAWS Trunk Angle (Hip vertex)
    # 180 = Straight Standing
    # 90 = Bent in half
    torso_angle = calculate_angle(shoulder, hip, knee)
    
    # 2. Calculate Knee Angle (for Squat vs Stoop context)
    knee_angle = calculate_angle(hip, knee, ankle)
    
    # --- EAWS STANDARDS LOGIC ---
    
    # ZONE 1: NEUTRAL (0° - 20° bend) -> Angle 160° - 180°
    if torso_angle > 160:
        return 0 # Safe
        
    # ZONE 2: CRITICAL / STRONGLY BENT (> 60° bend) -> Angle < 120°
    # Matches EAWS Class 4 "Strongly Bent Forward"
    if torso_angle < 120:
        return 2 # Critical
        
    # ZONE 3: WARNING / BENT (20° - 60° bend) -> Angle 120° - 160°
    # Matches EAWS Class 3 "Bent Forward"
    # Here we apply the NIOSH "Squat Check" to refine the warning.
    if 120 <= torso_angle <= 160:
        # If legs are straight (Stoop), this is worse than if knees are bent (Squat)
        if knee_angle > 150: 
            return 1 # Warning (Bad form: Stooping)
        else:
            return 0 # Safe-ish (Good form: Squatting slightly)
            
    return 0 # Default

def interpolate_sequence(timestamps, frames, target_timestamps):
    """Linearly interpolates a sequence of frames to match exact target timestamps."""
    N, V, C = frames.shape
    interpolated = np.zeros((len(target_timestamps), V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            interpolated[:, v, c] = np.interp(target_timestamps, timestamps, frames[:, v, c])
    return interpolated

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0 # fallback

    raw_frames = []
    raw_timestamps = []
    y_labels_raw = []
    frame_count = 0
    SKIP = 2 # Process every 2nd frame to speed up Mediapipe, but physics remains accurate

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % SKIP != 0: continue
        
        timestamp = frame_count / fps

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            lms = detection_result.pose_landmarks[0]
            label = get_bio_mechanical_label(lms)
            
            indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            skel_raw = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in indices])
            norm_skel = normalize_skeleton(skel_raw)
            
            raw_frames.append(norm_skel)
            raw_timestamps.append(timestamp)
            y_labels_raw.append(label)

    cap.release()
    
    if len(raw_frames) < 10:
        return [], []

    raw_frames = np.array(raw_frames)
    raw_timestamps = np.array(raw_timestamps)
    
    # --- TEMPORAL INTERPOLATION (FPS-Agnostic) ---
    X_seq, y_seq = [], []
    
    # We want exactly 50 frames representing exactly 5.0 seconds of real-world motion.
    SEQ_LEN = 50
    WINDOW_DURATION = 5.0 
    DT = WINDOW_DURATION / SEQ_LEN # 0.1 seconds per frame
    
    # Slide a 5-second window over the video
    start_time = raw_timestamps[0]
    end_time = raw_timestamps[-1]
    
    current_t = start_time
    while current_t + WINDOW_DURATION <= end_time:
        window_start = current_t
        window_end = current_t + WINDOW_DURATION
        
        # Get all frames that fall within this 5.0s window
        mask = (raw_timestamps >= window_start) & (raw_timestamps <= window_end)
        window_times = raw_timestamps[mask]
        window_frames = raw_frames[mask]
        window_labels = [y_labels_raw[i] for i, m in enumerate(mask) if m]
        
        # We need at least 10 valid points to interpolate safely
        if len(window_times) >= 10:
            target_times = np.linspace(window_start, window_end, SEQ_LEN)
            
            # Interpolate to exactly 50 perfectly-spaced frames
            S_interp = interpolate_sequence(window_times, window_frames, target_times)
            
            # True Physics Velocity: (p2 - p1) / dt
            V_interp = np.diff(S_interp, axis=0, prepend=S_interp[0:1]) / DT
            
            # Combine
            combined = np.concatenate((S_interp, V_interp), axis=2) # (50, 17, 6)
            
            # Label
            crit_count = window_labels.count(2)
            bad_count = window_labels.count(1)
            threshold = len(window_labels) * 0.2
            
            if crit_count > threshold: label = 2
            elif bad_count > threshold: label = 1
            else: label = 0
            
            X_seq.append(combined.reshape(SEQ_LEN, -1)) # (50, 102)
            y_seq.append(label)
            
        current_t += 1.0 # 1 second stride

    return X_seq, y_seq
if __name__ == "__main__":
    search = os.path.join(DATA_ROOT, "**", "*.mp4")
    files = glob.glob(search, recursive=True)
    logging.info(f"Found {len(files)} videos. Starting Bio-Mechanical Processing...")
    
    all_X, all_y = [], []
    for i, f in enumerate(files):
        logging.info(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
        x, y = process_video(f)
        all_X.extend(x)
        all_y.extend(y)
        
    if all_X:
        os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)
        np.save(OUTPUT_X, np.array(all_X))
        np.save(OUTPUT_Y, np.array(all_y))
        logging.info("SUCCESS! Nuanced Dataset Created.")