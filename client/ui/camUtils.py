#camUtils.py
import cv2

def find_available_cameras(max_to_check=5):
    """
    Checks for available camera indices up to 'max_to_check'.
    
    Returns:
        A list of strings (e.g., "Camera 0") for available cameras.
    """
    available_cameras = []
    for i in range(max_to_check):
        # Force V4L2 backend to avoid FFMPEG buffering
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            available_cameras.append(f"Camera {i}")
            cap.release()
            
    if not available_cameras:
        available_cameras.append("No Cameras Found")
        
    return available_cameras
