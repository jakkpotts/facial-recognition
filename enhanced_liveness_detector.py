import cv2
import numpy as np
from collections import deque
import dlib
from scipy.spatial import distance
import time
from typing import Dict, List, Tuple, Optional
import mediapipe as mp

class EnhancedLivenessDetector:
    def __init__(self, 
                 blink_threshold: float = 0.3,
                 blink_consec_frames: int = 3,
                 movement_threshold: float = 1.5,
                 texture_threshold: float = 30):
        """
        Enhanced liveness detector with multiple anti-spoofing techniques.
        """
        # Core detectors initialization
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Blink detection parameters
        self.blink_threshold = blink_threshold
        self.blink_consec_frames = blink_consec_frames
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        
        # Movement detection parameters
        self.head_positions = deque(maxlen=30)
        self.movement_threshold = movement_threshold
        self.last_head_pose = None
        
        # Texture analysis parameters
        self.texture_threshold = texture_threshold
        
        # Depth estimation parameters
        self.depth_history = deque(maxlen=10)
        
        # Face mesh landmarks for specific regions
        self.LEFT_EYE_INDICES = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
        self.RIGHT_EYE_INDICES = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
        self.FACE_OVAL_INDICES = list(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)
        
        # Challenge system
        self.challenge_active = False
        self.challenge_start_time = None
        self.challenge_type = None
        self.challenge_completed = False
        self.challenge_progress = 0
        self.required_progress = 3  # Number of successful detections needed

    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Calculate eye aspect ratio (EAR) for blink detection."""
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def detect_blink(self, landmarks: np.ndarray) -> bool:
        """
        Detect blink based on EAR and update blink counters.
        """
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < self.blink_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_consec_frames:
                self.total_blinks += 1
                self.last_blink_time = time.time()
                if self.challenge_active and self.challenge_type == 'blink_three_times':
                    self.challenge_progress += 1
            self.blink_counter = 0
        
        return time.time() - self.last_blink_time < 2.0

    def detect_head_turn(self, landmarks: np.ndarray, direction: str) -> bool:
        """
        Detect head turn in specified direction.
        """
        # Calculate face center
        face_center = np.mean(landmarks, axis=0)
        
        if self.last_head_pose is None:
            self.last_head_pose = face_center
            return False
        
        # Calculate horizontal movement
        movement = face_center[0] - self.last_head_pose[0]
        self.last_head_pose = face_center
        
        # Check movement direction
        if direction == 'left' and movement < -self.movement_threshold:
            self.challenge_progress += 1
            return True
        elif direction == 'right' and movement > self.movement_threshold:
            self.challenge_progress += 1
            return True
        
        return False

    def detect_eyebrow_raise(self, landmarks: np.ndarray) -> bool:
        """
        Detect eyebrow raising motion.
        """
        # Get eyebrow and eye positions
        left_eyebrow = np.mean(landmarks[17:22], axis=0)
        right_eyebrow = np.mean(landmarks[22:27], axis=0)
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Calculate vertical distances
        left_distance = abs(left_eyebrow[1] - left_eye[1])
        right_distance = abs(right_eyebrow[1] - right_eye[1])
        
        # Check if distance is larger than normal
        if left_distance > 20 and right_distance > 20:  # Threshold may need adjustment
            self.challenge_progress += 1
            return True
        return False

    def detect_micro_movement(self, landmarks: np.ndarray) -> bool:
        """
        Detect small natural head movements.
        """
        center_point = np.mean(landmarks, axis=0)
        self.head_positions.append(center_point)
        
        if len(self.head_positions) < 2:
            return False
        
        movement = np.linalg.norm(self.head_positions[-1] - self.head_positions[-2])
        return self.movement_threshold * 0.5 < movement < self.movement_threshold * 2

    def generate_challenge(self) -> Dict:
        """
        Generate a random liveness challenge.
        Returns a dictionary with challenge type and timeout.
        """
        challenges = [
            'blink_three_times',
            'turn_head_left',
            'turn_head_right',
            'raise_eyebrows'
        ]
        
        if not self.challenge_active:
            self.challenge_type = np.random.choice(challenges)
            self.challenge_active = True
            self.challenge_start_time = time.time()
            self.challenge_completed = False
            self.challenge_progress = 0
            
            return {
                'type': self.challenge_type,
                'timeout': 5.0,  # 5 seconds to complete challenge
                'required_progress': self.required_progress
            }
        
        return {
            'type': self.challenge_type,
            'timeout': max(0, 5.0 - (time.time() - self.challenge_start_time)),
            'progress': self.challenge_progress,
            'required_progress': self.required_progress
        }

    def verify_challenge_response(self, landmarks: np.ndarray) -> bool:
        """
        Verify if the current challenge has been completed.
        """
        if not self.challenge_active:
            return False
        
        # Check for timeout
        if time.time() - self.challenge_start_time > 5.0:
            self.challenge_active = False
            return False
        
        # Update challenge progress based on type
        if self.challenge_type == 'blink_three_times':
            self.detect_blink(landmarks)
        elif self.challenge_type == 'turn_head_left':
            self.detect_head_turn(landmarks, 'left')
        elif self.challenge_type == 'turn_head_right':
            self.detect_head_turn(landmarks, 'right')
        elif self.challenge_type == 'raise_eyebrows':
            self.detect_eyebrow_raise(landmarks)
        
        # Check if challenge is completed
        if self.challenge_progress >= self.required_progress:
            self.challenge_completed = True
            self.challenge_active = False
            return True
        
        return False

    def check_liveness(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Dict:
        """
        Enhanced liveness detection combining multiple techniques.
        """
        top, right, bottom, left = face_location
        face_roi = frame[top:bottom, left:right]
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = np.array([[p.x, p.y] for p in self.predictor(frame, rect).parts()])
        
        # Update challenge status
        if self.challenge_active:
            self.verify_challenge_response(landmarks)
        
        # Basic checks
        is_blink_pass = self.detect_blink(landmarks)
        is_movement_pass = self.detect_micro_movement(landmarks)
        
        # Calculate confidence score
        checks = [
            is_blink_pass,
            is_movement_pass,
        ]
        
        if self.challenge_completed:
            checks.append(True)  # Add weight for completed challenge
        
        confidence = round((sum(checks) / len(checks)) * 100)
        is_live = confidence >= 70.0
        
        return {
            'is_live': is_live,
            'confidence': confidence,
            'checks': {
                'blink': is_blink_pass,
                'micro_movement': is_movement_pass,
                'challenge_completed': self.challenge_completed
            },
            'challenge_status': self.generate_challenge() if self.challenge_active else None
        }

    def reset(self):
        """Reset all liveness detection states."""
        self.blink_counter = 0
        self.total_blinks = 0
        self.head_positions.clear()
        self.depth_history.clear()
        self.challenge_active = False
        self.challenge_completed = False
        self.challenge_type = None
        self.challenge_progress = 0
        self.last_head_pose = None