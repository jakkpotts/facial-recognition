import cv2
import time
from .camera_interface import CameraInterface

class MacBookCamera(CameraInterface):
    """Implementation for MacBook's built-in webcam"""
    def __init__(self):
        self.camera = None
        self.initialization_retries = 3
        self.frame_timeout = 2.0  # seconds
    
    def initialize(self):
        for attempt in range(self.initialization_retries):
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise RuntimeError("Could not open MacBook camera")
                
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, test_frame = self.camera.read()
                if not ret or test_frame is None:
                    raise RuntimeError("Camera opened but cannot capture frames")
                
                print("MacBook camera initialized successfully")
                return
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {str(e)}")
                if self.camera:
                    self.camera.release()
                if attempt < self.initialization_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError("Failed to initialize camera after multiple attempts")
    
    def capture_frame(self):
        if not self.camera or not self.camera.isOpened():
            raise RuntimeError("Camera is not initialized")
        
        start_time = time.time()
        while time.time() - start_time < self.frame_timeout:
            ret, frame = self.camera.read()
            if ret and frame is not None and frame.size > 0:
                if len(frame.shape) != 3:
                    raise ValueError("Invalid frame format: not a color image")
                return frame
            time.sleep(0.1)
        
        raise RuntimeError("Failed to capture valid frame within timeout period")
    
    def release(self):
        if self.camera:
            self.camera.release()
            self.camera = None
