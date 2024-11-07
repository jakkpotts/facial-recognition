import platform
import os

class CameraFactory:
    """Factory class to create appropriate camera instance based on platform"""
    @staticmethod
    def create_camera(preview=False):
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin":
            # For macOS (M2 chip)
            from .macbook_camera import MacBookCamera
            return MacBookCamera()
        elif system == "Linux" and os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model") as f:
                model = f.read()
                if "Raspberry Pi Zero 2" in model:
                    try:
                        from .arducam_camera import ArducamCamera
                        return ArducamCamera(preview=preview)
                    except ImportError:
                        print("Warning: ArducamCamera not available, falling back to alternative camera implementation.")
                        from .fallback_camera import FallbackCamera
                        return FallbackCamera(preview=preview)
        
        raise RuntimeError(f"Unsupported platform: {system} {machine}")
