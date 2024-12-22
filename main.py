import sys
from facial_recognition import FacialRecognition
import argparse

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

def main():
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    parser.add_argument('--image', type=str, help='Path to input image (if not using camera)')
    parser.add_argument('--known-faces-dir', type=str, default='known_faces',
                       help='Directory containing known faces (default: known_faces)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode without GUI')
    parser.add_argument('--disable-liveness', action='store_true',
                       help='Disable liveness detection')
    args = parser.parse_args()
    
    try:
        print("Starting Facial Recognition System...", flush=True)
        
        use_camera = args.image is None

        facial_recognition = FacialRecognition(
            known_faces_dir=args.known_faces_dir,
            use_camera=use_camera,
            input_image=args.image,
            headless=args.headless,
            disable_liveness=args.disable_liveness
        )
        facial_recognition.run()
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()