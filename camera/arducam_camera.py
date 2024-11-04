import time
import numpy as np
import subprocess
from picamera2 import Picamera2
from .camera_interface import CameraInterface

class ArducamCamera(CameraInterface):
    """Implementation for Arducam on Pi Zero 2 using Picamera2 with preview support"""
    def __init__(self, preview=False):
        self.camera = None
        self.initialization_retries = 3
        self.preview = preview
        self.preview_process = None

    def initialize(self):
        for attempt in range(self.initialization_retries):
            try:
                # Initialize Picamera2
                self.camera = Picamera2()
                
                # Configure camera
                config = self.camera.create_preview_configuration(
                    main={"size": (1280, 720),
                          "format": "RGB888"},
                    buffer_count=2
                )
                
                self.camera.configure(config)
                
                # Start the camera
                self.camera.start()
                
                # Start preview if requested
                if self.preview:
                    try:
                        # Kill any existing preview processes
                        subprocess.run(['pkill', 'libcamera-hello'], stderr=subprocess.DEVNULL)
                        # Start preview in background
                        self.preview_process = subprocess.Popen(
                            ['libcamera-hello', '-t', '0'],  # Run indefinitely
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        print("Preview started - use Ctrl+C to stop")
                    except Exception as e:
                        print(f"Warning: Could not start preview: {e}")
                        self.preview = False
                
                # Wait for camera to warm up
                time.sleep(0.5)
                
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
                        self.camera.close()
                    except:
                        pass
                if attempt < self.initialization_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Failed to initialize camera after {self.initialization_retries} attempts")

    def capture_frame(self):
        if not self.camera:
            raise RuntimeError("Camera is not initialized")
        
        try:
            # Capture frame using picamera2
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None

    def release(self):
        if self.preview_process:
            try:
                self.preview_process.terminate()
                self.preview_process.wait(timeout=1)
            except:
                try:
                    self.preview_process.kill()
                except:
                    pass
                
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                print(f"Error releasing camera: {str(e)}")
            finally:
                self.camera = None