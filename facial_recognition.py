import os
import time
import numpy as np
import face_recognition
import cv2
import json
import hashlib
import pickle
from datetime import datetime
from camera.camera_factory import CameraFactory

class FacialRecognition:
    def __init__(self, known_faces_dir="known_faces", use_camera=True, input_image=None):
        self.use_camera = use_camera
        self.input_image = input_image
        self.known_faces_dir = known_faces_dir
        self.cache_file = os.path.join(known_faces_dir, "encodings_cache.pkl")
        self.metadata_file = os.path.join(known_faces_dir, "encodings_metadata.json")
        
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
        self.load_known_faces(known_faces_dir)
        
        # Performance settings
        self.process_every_n_frames = 3
        self.frame_count = 0
        self.recognition_threshold = 0.6
        
        print("Facial Recognition System Initialized")
    
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
        print("Known people:", ', '.join(self.known_face_names))
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        # Convert from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Process each face found
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Get face distances to known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= self.recognition_threshold:
                    name = self.known_face_names[best_match_index]
                    confidence = round((1 - face_distances[best_match_index]) * 100, 1)
            
            # Draw rectangle and name with confidence
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            label = f"{name} ({confidence}%)" if name != "Unknown" else name
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Log recognition
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name != "Unknown":
                print(f"[{timestamp}] Detected: {name} (Confidence: {confidence}%)")
            else:
                print(f"[{timestamp}] Unknown Face Detected, No Match Found")
        
        return frame
    
    def _run_camera_mode(self):
        """Run facial recognition in camera mode"""
        print("Starting camera feed...")
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while True:
            try:
                frame = self.camera.capture_frame()
                consecutive_failures = 0  # Reset on successful capture
                
                if self.frame_count % self.process_every_n_frames == 0:
                    frame = self.process_frame(frame)
                
                cv2.imshow('Facial Recognition', frame)
                self.frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit command received")
                    break
                    
            except KeyboardInterrupt:
                print("\nStopping facial recognition...")
                break
                
            except Exception as e:
                consecutive_failures += 1
                print(f"Error capturing/processing frame: {str(e)}")
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Too many consecutive failures ({consecutive_failures}). Stopping.")
                    break
                time.sleep(0.5)  # Brief pause before retry
    
    def _run_image_mode(self):
        """Run facial recognition on a single image"""
        # Load and process the image
        frame = cv2.imread(self.input_image)
        if frame is None:
            raise ValueError(f"Could not load image: {self.input_image}")
        
        # Process the image
        processed_frame = self.process_frame(frame)
        
        # Display result and wait for key press
        cv2.imshow('Facial Recognition', processed_frame)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        
        # Save the processed image
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
                    
        except KeyboardInterrupt:
            print("\nStopping facial recognition...")
        finally:
            cv2.destroyAllWindows()
            if self.use_camera:
                self.camera.release()