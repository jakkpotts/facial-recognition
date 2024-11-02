import time
from .camera_interface import CameraInterface

class ArducamCamera(CameraInterface):
    """Implementation for Arducam on Pi Zero 2"""
    def __init__(self):
        try:
            from picamera2 import Picamera2
            self.Picamera2 = Picamera2
        except ImportError:
            raise ImportError("picamera2 module not found. Required for Arducam on Pi Zero 2")
        self.camera = None
        self.initialization_retries = 3
    
    def initialize(self):
        for attempt in range(self.initialization_retries):
            try:
                self.camera = self.Picamera2()
                config = self.camera.create_preview_configuration(main={"size": (1280, 720)})
                self.camera.configure(config)
                self.camera.start()
                
                test_frame = self.camera.capture_array()
                if test_frame is None or test_frame.size == 0:
                    raise RuntimeError("Camera started but cannot capture frames")
                
                print("Arducam camera initialized successfully")
                return
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {str(e)}")
                if self.camera:
                    try:
                        self.camera.stop()
                    except:
                        pass
                if attempt < self.initialization_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError("Failed to initialize camera after multiple attempts")
    
    def capture_frame(self):
        if not self.camera:
            raise RuntimeError("Camera is not initialized")
        
        frame = self.camera.capture_array()
        if frame is None or frame.size == 0:
            raise RuntimeError("Failed to capture valid frame")
        
        if len(frame.shape) != 3:
            raise ValueError("Invalid frame format: not a color image")
        
        return frame
    
    def release(self):
        if self.camera:
            self.camera.stop()
            self.camera = None
