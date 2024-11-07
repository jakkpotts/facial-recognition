import cv2

class FallbackCamera:
    def __init__(self, preview=False):
        self.capture = cv2.VideoCapture(0)
        self.preview = preview

    def initialize(self):
        pass

    def capture_frame(self):
        ret, frame = self.capture.read()
        return frame if ret else None

    def release(self):
        self.capture.release()
