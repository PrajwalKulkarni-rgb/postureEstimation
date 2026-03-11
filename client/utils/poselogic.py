import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import logging
import os
from collections import deque

from utils.normalization import normalize_skeleton
from utils.filtering import OneEuroFilter

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), 
    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

class SafetyLogic:
    def __init__(self, fps=30):
        self.fps = fps
        self.history = deque(maxlen=self.fps * 2) 
        self.cooldown = 0 

    def update(self, prediction_class):
        self.history.append(prediction_class)
        if self.cooldown > 0:
            self.cooldown -= 1
            return None 

        if len(self.history) >= 30:
            recent_window = list(self.history)[-30:]
            critical_count = recent_window.count(2)
            if critical_count > 20:
                self.cooldown = self.fps * 3
                return "CRITICAL"
            warning_count = recent_window.count(1)
            if warning_count > 20:
                self.cooldown = self.fps * 5 
                return "WARNING"
        return "SAFE"

class PoseLogic:
    def __init__(self):
        self.logger = logging.getLogger("PoseLogic")
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model/pose_landmarker_full.task'))
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.use_ai = False 
        self.SEQ_LEN = 50
        self.buffer = deque(maxlen=self.SEQ_LEN)
        self.filters = {} 
        self.safety_monitor = SafetyLogic(fps=30)
        self.status = "Initializing"
        self.color = (0, 255, 0) 

    def get_smoothed_landmarks(self, raw_landmarks):
        timestamp = time.time()
        smoothed = []
        for i, lm in enumerate(raw_landmarks):
            if i not in self.filters:
                self.filters[i] = {
                    'x': OneEuroFilter(timestamp, lm.x),
                    'y': OneEuroFilter(timestamp, lm.y),
                    'z': OneEuroFilter(timestamp, lm.z)
                }
            f = self.filters[i]
            s_x = f['x'](timestamp, lm.x)
            s_y = f['y'](timestamp, lm.y)
            s_z = f['z'](timestamp, lm.z)
            
            class SmoothPoint:
                def __init__(self, x, y, z, v):
                    self.x, self.y, self.z, self.visibility = x, y, z, v
            smoothed.append(SmoothPoint(s_x, s_y, s_z, lm.visibility))
        return smoothed

    def get_coco17_skeleton(self, landmarks):
        indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        points = []
        for i in indices:
            lm = landmarks[i]
            points.append([lm.x, lm.y, lm.z])
        return np.array(points)

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def process_frame(self, frame):

        annotated_img = frame.copy()
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            raw_lms = detection_result.pose_landmarks[0]
            lms = self.get_smoothed_landmarks(raw_lms)
            
            #Context Logic
            left_wrist_y = lms[15].y
            left_knee_y = lms[25].y
            is_lifting = left_wrist_y > left_knee_y

            #Buffer Logic
            skel = self.get_coco17_skeleton(lms)
            norm_skel = normalize_skeleton(skel)
            self.buffer.append(norm_skel)
            
            #geometric logic
            torso_angle = self.calculate_angle(lms[11], lms[23], lms[25])
            knee_angle = self.calculate_angle(lms[23], lms[25], lms[27])
            is_stooping = (torso_angle < 135) and (knee_angle > 150)
            
            #draw on copy
            h, w, _ = annotated_img.shape
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_img, (cx, cy), 3, (0, 0, 255), -1)
            
            for connection in POSE_CONNECTIONS:
                p1_idx, p2_idx = connection
                if p1_idx < len(lms) and p2_idx < len(lms):
                    p1 = lms[p1_idx]
                    p2 = lms[p2_idx]
                    if getattr(p1, 'visibility', 1.0) > 0.3 and getattr(p2, 'visibility', 1.0) > 0.3:
                        cx1, cy1 = int(p1.x * w), int(p1.y * h)
                        cx2, cy2 = int(p2.x * w), int(p2.y * h)
                        cv2.line(annotated_img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            
            cv2.rectangle(annotated_img, (0,0), (450, 80), self.color, -1)
            cv2.putText(annotated_img, self.status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            debug_text = f"Torso: {int(torso_angle)} | Lift: {is_lifting}"
            cv2.putText(annotated_img, debug_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
        return annotated_img