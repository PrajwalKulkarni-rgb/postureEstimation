import sys
import os
import cv2
import time
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QFrame, QSizePolicy, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette


from ui.style import STYLESHEET
from ui.camUtils import find_available_cameras
from ui.videoThread import CameraWorker
from ui.inference_worker import InferenceWorker

class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Safety Monitor Pro")
        self.resize(1400, 850)
        self.setStyleSheet(STYLESHEET)
        
        self.camera_thread = None
        self.inference_thread = None
        self.inference_worker = None
        self.is_recording = False
        self.video_writer = None
        
        self.init_ui()

    def init_ui(self):
        # Main Container
        central_widget = QWidget()
        central_widget.setObjectName("MainContainer")
        self.setCentralWidget(central_widget)
        
        # Main Layout (Vertical: Header -> Feeds -> Controls)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- 1. Header ---
        header = QLabel("REAL-TIME ERGONOMIC MONITORING SYSTEM")
        header.setObjectName("HeaderLabel")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # --- 2. Video Feeds (Horizontal Layout) ---
        feeds_layout = QHBoxLayout()
        feeds_layout.setSpacing(30)
        
        # Left: Raw Camera
        self.raw_container = self.create_feed_container("Raw Input Feed")
        self.raw_label = self.raw_container.findChild(QLabel, "FeedLabel")
        feeds_layout.addWidget(self.raw_container)
        
        # Right: AI Analysis (With Status Border)
        self.model_container = self.create_feed_container("AI Analysis Feed")
        self.model_label = self.model_container.findChild(QLabel, "FeedLabel")
        self.model_frame = self.model_container  # We will color this border
        feeds_layout.addWidget(self.model_container)
        
        main_layout.addLayout(feeds_layout, stretch=1)

        # --- 3. Alert Overlay (Status Bar) ---
        self.alert_bar = QLabel("SYSTEM READY")
        self.alert_bar.setObjectName("AlertBar_Safe") # Default Style
        self.alert_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_bar.setFixedHeight(60)
        main_layout.addWidget(self.alert_bar)

        # --- 4. Controls (Bottom Bar) ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        # Camera Selection
        self.combo = QComboBox()
        self.combo.addItems(find_available_cameras())
        self.combo.setFixedWidth(200)
        controls_layout.addWidget(QLabel("Source:"))
        controls_layout.addWidget(self.combo)
        
        # Buttons
        self.btn_start = QPushButton("▶ START SYSTEM")
        self.btn_start.clicked.connect(self.start_system)
        self.btn_start.setObjectName("BtnStart")
        controls_layout.addWidget(self.btn_start)
        
        self.btn_record = QPushButton("● REC")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setObjectName("BtnRecord")
        self.btn_record.setEnabled(False) # Enable only when running
        controls_layout.addWidget(self.btn_record)
        
        self.btn_stop = QPushButton("⏹ STOP")
        self.btn_stop.clicked.connect(self.stop_system)
        self.btn_stop.setObjectName("BtnStop")
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)

        controls_layout.addStretch() # Push exit button to right
        
        self.btn_exit = QPushButton("✖ EXIT")
        self.btn_exit.setObjectName("BtnExit")
        self.btn_exit.clicked.connect(self.close)
        controls_layout.addWidget(self.btn_exit)
        
        main_layout.addLayout(controls_layout)

    def create_feed_container(self, title):
        """Creates a styled frame for video feeds."""
        frame = QFrame()
        frame.setObjectName("FeedFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        title_lbl = QLabel(title)
        title_lbl.setObjectName("FeedTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        feed_lbl = QLabel()
        feed_lbl.setObjectName("FeedLabel")
        feed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feed_lbl.setStyleSheet("background-color: #000;")
        feed_lbl.setMinimumSize(640, 480)
        
        layout.addWidget(title_lbl)
        layout.addWidget(feed_lbl)
        return frame

    # --- LOGIC ---

    def start_system(self):
        cam_text = self.combo.currentText()
        if "No Cameras" in cam_text: return
        idx = int(cam_text.split()[-1])
        
        # Init Workers
        self.camera_thread = CameraWorker(idx)
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)
        
        # Connect Signals
        self.camera_thread.frame_captured.connect(self.inference_worker.process_frame)
        
        self.inference_worker.update_raw_feed.connect(self.set_raw)
        
        # RECORDING HOOK: Connect to the Analysis Feed
        self.inference_worker.update_model_feed.connect(self.set_model)
        self.inference_worker.update_model_feed.connect(self.handle_recording_qimage)
        
        self.inference_worker.server_status.connect(self.update_status_ui)
        
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.inference_thread.finished.connect(self.inference_thread.deleteLater)
        
        self.inference_thread.start()
        self.camera_thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.combo.setEnabled(False)
        self.update_status_ui("Connecting...")

    def stop_system(self):
        if self.is_recording:
            self.toggle_recording() 
            
        try:
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread.wait()
        except RuntimeError:
            pass # Object already deleted by deleteLater
            
        if self.inference_worker:
            self.inference_worker.stop()
            
        try:
            if self.inference_thread:
                self.inference_thread.quit()
                self.inference_thread.wait()
        except RuntimeError:
            pass # Object already deleted by deleteLater
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.combo.setEnabled(True)
        self.model_frame.setStyleSheet("#FeedFrame { border: 2px solid #444; }")
        self.alert_bar.setText("SYSTEM STOPPED")
        self.alert_bar.setObjectName("AlertBar_Safe")
        self.refresh_style()

    def update_status_ui(self, status_msg):
        if "CRITICAL" in status_msg:
            color = "#FF0000" 
            style_id = "AlertBar_Critical"
            text = "⚠️ CRITICAL RISK DETECTED"
        elif "Warning" in status_msg:
            color = "#FF8800"
            style_id = "AlertBar_Warning"
            text = "⚠️ POSTURE WARNING"
        elif "Safe" in status_msg:
            color = "#00FF00"
            style_id = "AlertBar_Safe"
            text = "✅ WORKER SAFE"
        else:
            color = "#444"
            style_id = "AlertBar_Safe"
            text = status_msg 

        self.model_frame.setStyleSheet(f"#FeedFrame {{ border: 4px solid {color}; border-radius: 10px; }}")
        
        self.alert_bar.setText(text)
        self.alert_bar.setObjectName(style_id)
        self.refresh_style() 

    def refresh_style(self):
        self.style().unpolish(self.alert_bar)
        self.style().polish(self.alert_bar)

    def set_raw(self, img): 
        self.raw_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.raw_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def set_model(self, img): 
        self.model_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.model_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def qimage_to_cv2(self, qimage):
        """Converts QImage back to OpenCV format (numpy array) for recording."""
        # Convert to ARGB32 first to ensure standard byte order
        qimage = qimage.convertToFormat(QImage.Format.Format_ARGB32)
        
        width = qimage.width()
        height = qimage.height()
        
        # Get pointer to bits
        ptr = qimage.bits()
        # CRITICAL: Some PyQt versions require this check
        if ptr is None: return None
        
        ptr.setsize(qimage.sizeInBytes())
        
        # Convert to numpy array
        arr = np.array(ptr).reshape(height, width, 4)  # 4 channels (BGRA)
        
        # Convert BGRA (Qt) to BGR (OpenCV)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def toggle_recording(self):
        if not self.is_recording:
            os.makedirs("recordings", exist_ok=True)
            filename = f"recordings/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            
            # Initialize VideoWriter
            # We use a standard size (640x480) for simplicity, or you can dynamically detect
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            
            self.is_recording = True
            self.btn_record.setText("⏹ STOP REC")
            self.btn_record.setStyleSheet("background-color: #FF0000; color: white;")
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.btn_record.setText("● REC")
            self.btn_record.setStyleSheet("") 

    def handle_recording_qimage(self, qimage):
        if self.is_recording and self.video_writer:
            frame = self.qimage_to_cv2(qimage)
            if frame is not None:
                # Resize to match the initialized writer size
                if frame.shape[:2] != (480, 640):
                    frame = cv2.resize(frame, (640, 480))
                self.video_writer.write(frame)

    def closeEvent(self, event):
        self.stop_system()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernWindow()
    window.show()
    sys.exit(app.exec())