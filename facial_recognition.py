import os
import time
import numpy as np
import face_recognition
import cv2
import json
import hashlib
import pickle
import signal
from datetime import datetime

from camera.camera_factory import CameraFactory
from enhanced_liveness_detector import EnhancedLivenessDetector

class FacialRecognition:
    def __init__(self, known_faces_dir="known_faces", use_camera=True, input_image=None, headless=False, disable_liveness=False):
        self.use_camera = use_camera
        self.input_image = input_image
        self.known_faces_dir = known_faces_dir
        self.headless = headless
        self.disable_liveness = disable_liveness
        self.running = True
        self.cache_file = os.path.join(known_faces_dir, "encodings_cache.pkl")
        self.metadata_file = os.path.join(known_faces_dir, "encodings_metadata.json")
        self.liveness_detector = EnhancedLivenessDetector()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Performance settings
        self.process_every_n_frames = 6  # increase for slower systems
        self.frame_count = 0
        self.recognition_threshold = 0.6
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        self.display_scale = 1  # decrease for slower systems
        
        # Challenge tracking
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_display_duration = 15  # seconds to display challenge message
        self.challenge_interval = 20  # seconds between challenges
        self.last_challenge_time = time.time() - self.challenge_interval  # Allow immediate first challenge
        
        # Initialize camera if needed
        if self.use_camera:
            try:
                self.camera = CameraFactory.create_camera(preview = True) if not headless else CameraFactory.create_camera(preview = False) 
                self.camera.initialize()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize camera: {str(e)}")
        elif not os.path.exists(input_image):
            raise FileNotFoundError(f"Input image not found: {input_image}")
        
        # Initialize known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(self.known_faces_dir)
        
        print("Facial Recognition System Initialized")
        
        # Add face detection results cache
        self.last_face_results = []
        self.results_expiry = 0.5  # How long to keep showing the same results (in seconds)
        self.last_detection_time = 0
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\nShutdown signal received. Cleaning up...", flush=True)
        self.running = False
    
    def _calculate_image_hash(self, image_path):
        """Calculate SHA-256 hash of an image file"""
        hasher = hashlib.sha256()
        with open(image_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def _get_directory_metadata(self, base_directory):
        """Get metadata for all images in the known faces directory"""
        metadata = {}
        for person_name in os.listdir(base_directory):
            person_dir = os.path.join(base_directory, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            for image_file in os.listdir(person_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, image_file)
                    metadata[image_path] = {
                        'hash': self._calculate_image_hash(image_path),
                        'modified_time': os.path.getmtime(image_path)
                    }
        return metadata
    
    def _should_rebuild_cache(self, base_directory):
        """Check if cache needs to be rebuilt by comparing metadata"""
        if not os.path.exists(self.cache_file) or not os.path.exists(self.metadata_file):
            return True, {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                old_metadata = json.load(f)
            
            current_metadata = self._get_directory_metadata(base_directory)
            
            # Instead of rebuilding everything, identify only changed files
            changed_files = {}
            for path, current_data in current_metadata.items():
                if path not in old_metadata or \
                   current_data['hash'] != old_metadata[path]['hash'] or \
                   current_data['modified_time'] != old_metadata[path]['modified_time']:
                    changed_files[path] = current_data
            
            return len(changed_files) > 0, changed_files
            
        except Exception as e:
            print(f"Error checking cache: {e}")
            return True, {}
    
    def load_known_faces(self, base_directory):
        """Load known faces from directory structure with caching"""
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)
            print(f"Created {base_directory} - please add person directories containing face images")
            return
        
        needs_update, changed_files = self._should_rebuild_cache(base_directory)
        
        # Load existing cache if available
        if os.path.exists(self.cache_file) and os.path.exists(self.metadata_file):
            try:
                print("Loading existing face encodings from cache...")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.known_face_encodings = cache_data['encodings']
                    self.known_face_names = cache_data['names']
                
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                if not needs_update:
                    print(f"Cache is up to date with {len(self.known_face_names)} people")
                    if self.known_face_names:
                        print("Known people:", ', '.join(self.known_face_names))
                    return
                    
            except Exception as e:
                print(f"Error loading cache: {e}")
                needs_update = True
                changed_files = self._get_directory_metadata(base_directory)
                self.known_face_encodings = []
                self.known_face_names = []
                metadata = {}

        # Process only new or modified files
        if needs_update:
            print("Processing new or modified faces...")
            for image_path in changed_files:
                person_name = os.path.basename(os.path.dirname(image_path))
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        # If person already exists, update their encoding
                        if person_name in self.known_face_names:
                            idx = self.known_face_names.index(person_name)
                            self.known_face_encodings[idx] = encodings[0]
                        else:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_name)
                        print(f"  ✓ Successfully processed {os.path.basename(image_path)}")
                        metadata[image_path] = changed_files[image_path]
                    else:
                        print(f"  ✗ No face found in {os.path.basename(image_path)}")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {os.path.basename(image_path)}: {e}")

            # Save updated cache and metadata
            try:
                cache_data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f)
                
                print("\nCache updated successfully")
            except Exception as e:
                print(f"Error saving cache: {e}")

        print(f"\nTotal people loaded: {len(self.known_face_names)}")
        if self.known_face_names:
            print("Known people:", ', '.join(self.known_face_names))

    def process_frame(self, frame):
        current_time = time.time()
        results_expired = (current_time - self.last_detection_time) > self.results_expiry
        
        # Only process face detection on every Nth frame or if results expired
        if self.frame_count % self.process_every_n_frames == 0 or results_expired:
            # Convert from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            # Scale face locations back up
            face_locations = [(int(top * 2), int(right * 2), int(bottom * 2), int(left * 2)) 
                            for top, right, bottom, left in face_locations]
            
            # Process faces and store results
            self.last_face_results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                result = {
                    'location': (top, right, bottom, left),
                    'encoding': face_encoding,
                    'name': "Unknown",
                    'confidence': 0,
                    'liveness_result': None
                }
                
                # Check liveness if not disabled
                if not self.disable_liveness:
                    result['liveness_result'] = self.liveness_detector.check_liveness(frame, (top, right, bottom, left))
                else:
                    result['liveness_result'] = {'is_live': True, 'confidence': 100, 'checks': {'challenge_completed': True}}
                
                # Get face distances to known faces
                if len(self.known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] <= self.recognition_threshold:
                        result['name'] = self.known_face_names[best_match_index]
                        result['confidence'] = round((1 - face_distances[best_match_index]) * 100, 1)
                
                self.last_face_results.append(result)
            
            self.last_detection_time = current_time
        
        # Always draw the last known results
        for result in self.last_face_results:
            top, right, bottom, left = result['location']
            liveness_result = result['liveness_result']
            
            # Determine box color based on liveness and challenge completion
            if self.disable_liveness or liveness_result['checks'].get('challenge_completed', False):
                box_color = (0, 255, 0)  # Green
            elif liveness_result['is_live']:
                box_color = (255, 165, 0)  # Orange
            else:
                box_color = (0, 0, 255)  # Red
            
            # Draw rectangle and labels
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Create background for text
            cv2.rectangle(frame, (left, bottom - 60), (right, bottom), box_color, cv2.FILLED)
            
            # Add name and confidence
            label = f"{result['name']} ({result['confidence']}%)" if result['name'] != "Unknown" else result['name']
            cv2.putText(frame, label, (left + 6, bottom - 36),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Add liveness confidence if not disabled
            if not self.disable_liveness:
                liveness_label = f"Liveness: {liveness_result['confidence']}%"
                cv2.putText(frame, liveness_label, (left + 6, bottom - 10),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Scale down display frame if needed
        if not self.headless and self.display_scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=self.display_scale, fy=self.display_scale)
        
        return frame
    
    def _run_camera_mode(self):
        """Run facial recognition in camera mode"""
        print("Starting camera feed...", flush=True)
        consecutive_failures = 0
        max_consecutive_failures = 5
        last_frame_time = time.time()
        
        while self.running:
            try:
                frame = self.camera.capture_frame()
                consecutive_failures = 0
                
                # Process frame (will only do face detection every Nth frame)
                frame = self.process_frame(frame)
                
                if not self.headless:
                    try:
                        cv2.imshow('Facial Recognition', frame)
                        # Use a shorter wait time
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Quit command received", flush=True)
                            break
                    except Exception as e:
                        print(f"Warning: Display error (running headless): {e}", flush=True)
                        self.headless = True
                
                self.frame_count += 1
                
                # More precise frame timing
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_time:
                    time.sleep(max(0, self.frame_time - elapsed))
                last_frame_time = time.time()
                
            except KeyboardInterrupt:
                print("\nStopping facial recognition...", flush=True)
                break
                
            except Exception as e:
                consecutive_failures += 1
                print(f"Error capturing/processing frame: {str(e)}", flush=True)
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Too many consecutive failures ({consecutive_failures}). Stopping.", flush=True)
                    break
                time.sleep(0.5)  # Brief pause before retry
        
        self._cleanup()

    def _cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...", flush=True)
        if hasattr(self, 'camera'):
            try:
                self.camera.release()
            except Exception as e:
                print(f"Error releasing camera: {e}", flush=True)
        
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error closing windows: {e}", flush=True)
        
        print("Cleanup complete", flush=True)
        
          
    def _run_image_mode(self):
        """Run facial recognition on a single image"""
        frame = cv2.imread(self.input_image)
        if frame is None:
            raise ValueError(f"Could not load image: {self.input_image}")
        
        processed_frame = self.process_frame(frame)
        
        cv2.imshow('Facial Recognition', processed_frame)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        
        output_path = f"processed_{os.path.basename(self.input_image)}"
        cv2.imwrite(output_path, processed_frame)
        print(f"\nProcessed image saved as: {output_path}")
    
    def run(self):
        """Main loop for facial recognition"""
        try:
            if self.use_camera:
                self._run_camera_mode()
            else:
                self._run_image_mode()
        except Exception as e:
            print(f"Error in main loop: {str(e)}", flush=True)
        finally:
            self._cleanup()