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
    def __init__(self, known_faces_dir="known_faces", use_camera=True, input_image=None):
        self.use_camera = use_camera
        self.input_image = input_image
        self.known_faces_dir = known_faces_dir
        self.headless = headless
        self.running = True
        self.cache_file = os.path.join(known_faces_dir, "encodings_cache.pkl")
        self.metadata_file = os.path.join(known_faces_dir, "encodings_metadata.json")
        self.liveness_detector = EnhancedLivenessDetector()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Performance settings
        self.process_every_n_frames = 2  # Process every other frame
        self.frame_count = 0
        self.recognition_threshold = 0.6
        self.target_fps = 10
        self.frame_time = 1.0 / self.target_fps
        
        # Challenge tracking
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_display_duration = 15  # seconds to display challenge message
        self.challenge_interval = 20  # seconds between challenges
        self.last_challenge_time = time.time() - self.challenge_interval  # Allow immediate first challenge
        
        # Initialize camera if needed
        if self.use_camera:
            try:
                self.camera = CameraFactory.create_camera()
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
            return True
            
        try:
            with open(self.metadata_file, 'r') as f:
                old_metadata = json.load(f)
            
            current_metadata = self._get_directory_metadata(base_directory)
            
            # Check if any files were added or removed
            if set(old_metadata.keys()) != set(current_metadata.keys()):
                return True
            
            # Check if any files were modified
            for path, current_data in current_metadata.items():
                if path not in old_metadata:
                    return True
                if current_data['hash'] != old_metadata[path]['hash']:
                    return True
                if current_data['modified_time'] != old_metadata[path]['modified_time']:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking cache: {e}")
            return True
    
    def load_known_faces(self, base_directory):
        """Load known faces from directory structure with caching"""
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)
            print(f"Created {base_directory} - please add person directories containing face images")
            return
        
        if not self._should_rebuild_cache(base_directory):
            try:
                print("Loading face encodings from cache...")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.known_face_encodings = cache_data['encodings']
                    self.known_face_names = cache_data['names']
                print(f"Successfully loaded {len(self.known_face_names)} people from cache")
                if self.known_face_names:
                    print("Known people:", ', '.join(self.known_face_names))
                return
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Rebuilding cache...")
        
        # Get all subdirectories (one per person)
        person_dirs = [d for d in os.listdir(base_directory) 
                      if os.path.isdir(os.path.join(base_directory, d))]
        
        if not person_dirs:
            print("No person directories found. Create directories named after each person and add their photos.")
            return
        
        metadata = {}
        
        # Process each person's directory
        for person_name in person_dirs:
            person_dir_path = os.path.join(base_directory, person_name)
            encodings_for_person = []
            
            # Valid image extensions
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            # Process each image in person's directory
            image_files = [f for f in os.listdir(person_dir_path) 
                         if f.lower().endswith(valid_extensions)]
            
            if not image_files:
                print(f"No valid images found for {person_name}")
                continue
                
            print(f"\nProcessing images for {person_name}:")
            for image_file in image_files:
                image_path = os.path.join(person_dir_path, image_file)
                try:
                    # Load and encode face
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        encodings_for_person.extend(encodings)
                        print(f"  ✓ Successfully processed {image_file}")
                        
                        # Store metadata for cache validation
                        metadata[image_path] = {
                            'hash': self._calculate_image_hash(image_path),
                            'modified_time': os.path.getmtime(image_path)
                        }
                    else:
                        print(f"  ✗ No face found in {image_file}")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {image_file}: {e}")
            
            # If we found any valid faces for this person, add them to our known faces
            if encodings_for_person:
                # Average all encodings for this person
                avg_encoding = np.mean(encodings_for_person, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(person_name)
                print(f"Added {person_name} with {len(encodings_for_person)} images")
            else:
                print(f"No valid face encodings found for {person_name}")
        
        # Save the cache and metadata
        if self.known_face_encodings:
            try:
                cache_data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f)
                
                print("\nCache saved successfully")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        print(f"\nTotal people loaded: {len(self.known_face_names)}")
        if self.known_face_names:
            print("Known people:", ', '.join(self.known_face_names))

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition with liveness challenges"""
        # Convert from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Challenge management
        current_time = time.time()
        
        # Generate new challenge if needed
        if (not self.liveness_detector.challenge_active and 
            current_time - self.last_challenge_time >= self.challenge_interval):
            self.current_challenge = self.liveness_detector.generate_challenge()
            self.challenge_start_time = current_time
            self.last_challenge_time = current_time
            print(f"\nNew Challenge: {self.current_challenge['type'].replace('_', ' ').title()}")
        
        # Display challenge message on frame if within display duration
        if (self.challenge_start_time and 
            current_time - self.challenge_start_time < self.challenge_display_duration):
            challenge_text = f"Please {self.current_challenge['type'].replace('_', ' ')}"
            cv2.putText(frame, challenge_text, (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check liveness
            liveness_result = self.liveness_detector.check_liveness(frame, (top, right, bottom, left))
            liveness_confidence = liveness_result['confidence']
            
            # Get face distances to known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = 0

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= self.recognition_threshold:
                    name = self.known_face_names[best_match_index]
                    confidence = round((1 - face_distances[best_match_index]) * 100, 1)

            # Determine box color based on liveness and challenge completion
            if liveness_result['checks'].get('challenge_completed', False):
                box_color = (0, 255, 0)  # Green for challenge completed
            elif liveness_result['is_live']:
                box_color = (255, 165, 0)  # Orange for live but challenge not completed
            else:
                box_color = (0, 0, 255)  # Red for not live

            # Draw rectangle and labels
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Create background for text
            cv2.rectangle(frame, (left, bottom - 60), (right, bottom), box_color, cv2.FILLED)
            
            # Add name and confidence
            label = f"{name} ({confidence}%)" if name != "Unknown" else name
            cv2.putText(frame, label, (left + 6, bottom - 36),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Add liveness confidence
            liveness_label = f"Liveness: {liveness_confidence}%"
            cv2.putText(frame, liveness_label, (left + 6, bottom - 10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Log recognition with enhanced information
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name != "Unknown":
                status = "✓ Challenge Complete" if liveness_result['checks'].get('challenge_completed', False) else "⋯ Challenge Pending"
                print(f"[{timestamp}] Detected: {name} (Confidence: {confidence}%) –– "
                      f"Liveness: {liveness_confidence}% –– {status}")
                
                # Print detailed liveness checks if confidence is low or verification fails
                if liveness_confidence < 75:
                    print("Liveness Check Details:")
                    for check, result in liveness_result['checks'].items():
                        print(f"  - {check.replace('_', ' ').title()}: {'Pass' if result else 'Fail'}")
            else:
                print(f"[{timestamp}] Unknown Face Detected, No Match Found")
            
        return frame
    
    def _run_camera_mode(self):
        """Run facial recognition in camera mode"""
        print("Starting camera feed...", flush=True)
        consecutive_failures = 0
        max_consecutive_failures = 5
        last_frame_time = time.time()
        
        while self.running:
            try:
                # Maintain target frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
                
                frame = self.camera.capture_frame()
                consecutive_failures = 0  # Reset on successful capture
                
                if self.frame_count % self.process_every_n_frames == 0:
                    frame = self.process_frame(frame)
                    
                    if not self.headless:
                        try:
                            cv2.imshow('Facial Recognition', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("Quit command received", flush=True)
                                break
                        except Exception as e:
                            print(f"Warning: Display error (running headless): {e}", flush=True)
                            self.headless = True
                
                self.frame_count += 1
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