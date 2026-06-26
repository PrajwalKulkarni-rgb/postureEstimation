import cv2
import numpy as np
import json
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from utils.poselogic import PoseLogic
from utils.network_client import NetworkClient

SERVER_URL = "ws://localhost:8000/ws/predict"

class InferenceWorker(QObject):
    update_raw_feed = pyqtSignal(QImage)
    update_model_feed = pyqtSignal(QImage)
    server_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logic = PoseLogic()
        self.client = NetworkClient(SERVER_URL)
        self.client.connect()
        self.is_busy = False

    @pyqtSlot(np.ndarray)
    def process_frame(self, frame):
        # Drop frame if we are still processing the previous one (prevents smooth queue delay)
        if self.is_busy:
            return
        
        self.is_busy = True
        try:
            # 1. IMMEDIATE COPY for Raw Feed
            # This isolates the raw video so nothing can draw on it
            raw_frame = frame.copy()
            
            # 2. SEPARATE COPY for Processing
            # We will draw lines on this one
            processing_frame = frame.copy() 
    
            # --- Server Communication ---
            server_pred = self.client.get_latest_prediction()
            
            if self.client.is_connected():
                self.server_status.emit("Online")
            else:
                self.server_status.emit("Offline")
    
            # Update Logic State
            if server_pred == 2:
                self.logic.status = "CRITICAL (Server)"
                self.logic.color = (0, 0, 255)
            elif server_pred == 1:
                self.logic.status = "Warning (Server)"
                self.logic.color = (0, 165, 255)
            elif server_pred == 0:
                self.logic.status = "Safe (Server)"
                self.logic.color = (0, 255, 0)
            else:
                # server_pred is None (Offline). Clear 'Server' tag so poselogic takes over.
                self.logic.status = "Edge"
    
            # Logic draws on 'processing_frame' and returns it
            annotated_frame = self.logic.process_frame(processing_frame)
    
            # Send to Server (Stream Single Frame)
            if len(self.logic.buffer) > 0:
                latest_frame = self.logic.buffer[-1] # (17, 3) numpy array
                self.client.send_skeleton(latest_frame)
    
            # Send Clean Raw Frame
            self.emit_image(raw_frame, self.update_raw_feed)
            
            # Send Annotated Frame
            self.emit_image(annotated_frame, self.update_model_feed)
        finally:
            self.is_busy = False

    def emit_image(self, cv_img, signal):
        """Helper to convert CV2 -> QImage"""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        
        # Create QImage from data
        temp_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        #MUST COPY() TO PERSIST IN MEMORY! 'rgb' is deleted by Python, and QImage points to garbage.
        final_img = temp_img.copy()
        
        signal.emit(final_img)

    def stop(self):
        self.client.close()
        