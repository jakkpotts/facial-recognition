from abc import ABC, abstractmethod

class CameraInterface(ABC):
    """Abstract base class for camera implementations"""
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def capture_frame(self):
        pass
    
    @abstractmethod
    def release(self):
        pass
