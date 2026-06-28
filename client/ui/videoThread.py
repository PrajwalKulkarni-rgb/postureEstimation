import cv2
import logging
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class CameraWorker(QThread):
    """
    Responsible only for capturing frames from the hardware.
    """
    # Emits a tuple of (timestamp, raw numpy array)
    frame_captured = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._is_running = True
        self.logger = logging.getLogger("CameraWorker")

    def run(self):
        import time
        # Force V4L2 backend
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        # Instruct V4L2 to only hold 1 frame in buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera {self.camera_index}")
            return

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                self.error_occurred.emit("Frame drop or camera disconnected")
                break
            
            # Emit the raw frame with its creation timestamp
            self.frame_captured.emit((time.time(), frame))
            self.msleep(1)

        cap.release()
        self.logger.info("Camera stopped")

    def stop(self):
        self._is_running = False