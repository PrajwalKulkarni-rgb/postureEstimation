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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    X_frames, y_labels = [], []
    frame_count = 0
    
    # Velocity Tracking
    prev_norm_skel = None
    SKIP = 3 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % SKIP != 0: continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            lms = detection_result.pose_landmarks[0]
            
            # 1. GET NUANCED LABEL
            label = get_bio_mechanical_label(lms)
            
            # 2. EXTRACT DATA
            indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            skel_raw = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in indices])
            
            # 3. NORMALIZE
            norm_skel = normalize_skeleton(skel_raw)
            
            # 4. VELOCITY
            if prev_norm_skel is None:
                velocity = np.zeros_like(norm_skel)
            else:
                velocity = norm_skel - prev_norm_skel
            prev_norm_skel = norm_skel
            
            # 5. STACK
            combined = np.concatenate((norm_skel, velocity), axis=1)
            X_frames.append(combined.flatten())
            y_labels.append(label)

    cap.release()
    
    # --- CHUNK INTO SEQUENCES PER VIDEO ---
    X_seq, y_seq = [], []
    seq_length = 50
    stride = 20
    
    for i in range(0, len(X_frames) - seq_length + 1, stride):
        clip = X_frames[i : i+seq_length]
        labels = y_labels[i : i+seq_length]
        
        # Robust Labeling: Majority vote with bias for critical classes
        crit_count = labels.count(2)
        bad_count = labels.count(1)
        threshold = seq_length * 0.2
        
        if crit_count > threshold: label = 2
        elif bad_count > threshold: label = 1
        else: label = 0
        
        X_seq.append(clip)
        y_seq.append(label)

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