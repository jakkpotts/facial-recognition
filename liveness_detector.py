import cv2
import numpy as np
from collections import deque
import dlib
from scipy.spatial import distance
import time

class LivenessDetector:
    def __init__(self):
        # Initialize facial landmark detector
        self.landmark_detector = dlib.get_frontal_face_detector()
        # Load pre-trained facial landmark predictor
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Blink detection parameters
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        self.blink_counter = 0
        self.blink_total = 0
        
        # Head movement detection
        self.head_positions = deque(maxlen=30)
        self.movement_threshold = 15.0
        
        # Texture analysis for printed photo detection
        self.texture_threshold = 25
        
        # Challenge-response system
        self.challenges = ['BLINK', 'TURN_LEFT', 'TURN_RIGHT', 'NOD']
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_timeout = 5.0  # seconds
        self.challenge_completed = False
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate the eye aspect ratio for blink detection"""
        # Compute vertical eye distances
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute horizontal eye distance
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _detect_blink(self, landmarks):
        """Detect if the person blinked"""
        # Extract eye regions
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Calculate eye aspect ratios
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratio
        ear = (left_ear + right_ear) / 2.0
        
        # Check if it's a blink
        if ear < self.EYE_AR_THRESH:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.blink_total += 1
            self.blink_counter = 0
        
        return self.blink_counter >= self.EYE_AR_CONSEC_FRAMES
    
    def _detect_head_movement(self, landmarks):
        """Detect head movement based on facial landmarks"""
        # Calculate center point of face
        center_point = np.mean(landmarks, axis=0)
        self.head_positions.append(center_point)
        
        if len(self.head_positions) < 2:
            return None
        
        # Calculate movement
        movement = np.linalg.norm(self.head_positions[-1] - self.head_positions[-2])
        
        # Determine direction if significant movement detected
        if movement > self.movement_threshold:
            dx = self.head_positions[-1][0] - self.head_positions[-2][0]
            dy = self.head_positions[-1][1] - self.head_positions[-2][1]
            
            if abs(dx) > abs(dy):
                return 'LEFT' if dx < 0 else 'RIGHT'
            else:
                return 'UP' if dy < 0 else 'DOWN'
        
        return None
    
    def _analyze_texture(self, frame, face_roi):
        """Detect if image is from a printed photo by analyzing texture patterns"""
        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply local binary pattern or other texture analysis
        # Here using simple variance as a basic texture measure
        blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        variance = np.var(blur)
        
        # Check if variance is too low (indicating printed photo)
        return variance > self.texture_threshold
    
    def _issue_challenge(self):
        """Issue a random liveness challenge"""
        if not self.current_challenge:
            self.current_challenge = np.random.choice(self.challenges)
            self.challenge_start_time = time.time()
            self.challenge_completed = False
            return self.current_challenge
        return None
    
    def _check_challenge_response(self, frame, landmarks):
        """Check if the current challenge has been completed"""
        if not self.current_challenge or time.time() - self.challenge_start_time > self.challenge_timeout:
            return False
        
        if self.current_challenge == 'BLINK':
            if self._detect_blink(landmarks):
                self.challenge_completed = True
        
        elif self.current_challenge in ['TURN_LEFT', 'TURN_RIGHT']:
            movement = self._detect_head_movement(landmarks)
            if movement == self.current_challenge:
                self.challenge_completed = True
        
        elif self.current_challenge == 'NOD':
            movement = self._detect_head_movement(landmarks)
            if movement == 'DOWN':
                self.challenge_completed = True
        
        return self.challenge_completed
    
    def check_liveness(self, frame, face_location):
        """Main liveness detection method"""
        # Extract face ROI
        top, right, bottom, left = face_location
        face_roi = frame[top:bottom, left:right]
        
        # Get facial landmarks
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = self.predictor(frame, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Perform various liveness checks
        results = {
            'is_live': False,
            'confidence': 0.0,
            'checks': {
                'texture': False,
                'blink_detected': False,
                'challenge_completed': False
            }
        }
        
        # 1. Check texture for printed photo detection
        results['checks']['texture'] = self._analyze_texture(frame, face_roi)
        
        # 2. Check for natural eye blinks
        results['checks']['blink_detected'] = self._detect_blink(landmarks)
        
        # 3. Challenge-response check
        if not self.current_challenge:
            self._issue_challenge()
        results['checks']['challenge_completed'] = self._check_challenge_response(frame, landmarks)
        
        # Calculate overall liveness confidence
        checks_passed = sum(results['checks'].values())
        results['confidence'] = (checks_passed / len(results['checks'])) * 100
        results['is_live'] = results['confidence'] >= 66.0  # At least 2/3 checks passed
        
        # Draw challenge instructions on frame
        if self.current_challenge and not self.challenge_completed:
            cv2.putText(frame, f"Please {self.current_challenge.replace('_', ' ')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return results

    def reset(self):
        """Reset all detection states"""
        self.blink_counter = 0
        self.blink_total = 0
        self.head_positions.clear()
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_completed = False