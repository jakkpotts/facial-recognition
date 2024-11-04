import time
import numpy as np
import libcamera
from libcamera import Transform
from .camera_interface import CameraInterface

class ArducamCamera(CameraInterface):
    """Implementation for Arducam on Pi Zero 2 using libcamera"""
    def __init__(self):
        self.camera = None
        self.camera_config = None
        self.stream = None
        self.initialization_retries = 3

    def initialize(self):
        for attempt in range(self.initialization_retries):
            try:
                self.camera = libcamera.Camera(libcamera.CameraManager().get_all_cameras()[0])
                self.camera_config = self.camera.generate_configuration(["main"])
                self.camera_config.transform = Transform(hflip=1, vflip=1)
                self.camera_config.main.size = libcamera.Size(1280, 720)
                self.camera_config.main.pixel_format = libcamera.PixelFormat("RGB888")
                self.camera.configure(self.camera_config)
                self.stream = self.camera_config.main.stream
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
                    except:
                        pass
                if attempt < self.initialization_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError("Failed to initialize camera after multiple attempts")

    def capture_frame(self):
        if not self.camera or not self.stream:
            raise RuntimeError("Camera is not initialized")

        buffer = self.stream.acquire_buffer()
        if buffer is None:
            raise RuntimeError("Failed to acquire buffer from camera stream")

        frame = np.array(buffer.planes[0])
        buffer.release()
        return frame

    def release(self):
        if self.camera:
            self.camera.stop()
            self.camera = None
            self.stream = None