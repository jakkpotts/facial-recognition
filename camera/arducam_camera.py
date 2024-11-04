import time
import numpy as np
import libcamera
from libcamera import Transform
from .camera_interface import CameraInterface

class ArducamCamera(CameraInterface):
    """Implementation for Arducam on Pi Zero 2 using libcamera"""
    def __init__(self):
        self.camera_manager = libcamera.CameraManager()
        self.camera = None
        self.camera_config = None
        self.stream = None
        self.initialization_retries = 3

    def initialize(self):
        for attempt in range(self.initialization_retries):
            try:
                cameras = self.camera_manager.cameras
                if not cameras:
                    raise RuntimeError("No cameras available")

                self.camera = cameras[0]
                self.camera.acquire()
                self.camera_config = self.camera.generate_configuration(["viewfinder"])
                self.camera_config.transform = Transform(hflip=1, vflip=1)
                self.camera_config[0].size = (1280, 720)
                self.camera_config[0].pixel_format = "RGB888"
                
                self.camera.configure(self.camera_config)
                self.camera.start()

                # Verify frame capture
                test_frame = self.capture_frame()
                if test_frame is None or test_frame.size == 0:
                    raise RuntimeError("Camera started but cannot capture frames")

                print("Arducam camera initialized successfully")
                return
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {str(e)}")
                if self.camera:
                    try:
                        self.camera.stop()
                        self.camera.release()
                    except:
                        pass
                if attempt < self.initialization_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError("Failed to initialize camera after multiple attempts")

    def capture_frame(self):
        if not self.camera or not self.stream:
            raise RuntimeError("Camera is not initialized")

        request = self.camera.capture_request()
        buffer = request.buffers[self.stream]
        data = np.array(buffer.planes[0].data, dtype=np.uint8).reshape((720, 1280, 3))  # Adjust size if necessary
        request.release()
        return data

    def release(self):
        if self.camera:
            self.camera.stop()
            self.camera.release()
            self.camera = None
            self.stream = None
