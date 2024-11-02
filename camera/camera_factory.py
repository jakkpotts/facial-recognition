import platform
import os
from .macbook_camera import MacBookCamera
from .arducam_camera import ArducamCamera

class CameraFactory:
    """Factory class to create appropriate camera instance based on platform"""
    @staticmethod
    def create_camera():
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin":
            return MacBookCamera()
        elif system == "Linux" and os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model") as f:
                model = f.read()
                if "Raspberry Pi Zero 2" in model:
                    return ArducamCamera()
        
        raise RuntimeError(f"Unsupported platform: {system} {machine}")
