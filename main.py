from facial_recognition import FacialRecognition
import argparse

def main():
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    parser.add_argument('--image', type=str, help='Path to input image (if not using camera)')
    parser.add_argument('--known-faces-dir', type=str, default='known_faces',
                       help='Directory containing known faces (default: known_faces)')
    args = parser.parse_args()
    
    try:
        use_camera = args.image is None
        facial_recognition = FacialRecognition(
            known_faces_dir=args.known_faces_dir,
            use_camera=use_camera,
            input_image=args.image
        )
        facial_recognition.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
