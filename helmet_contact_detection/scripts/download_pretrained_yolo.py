#!/usr/bin/env python3
"""
Download a pretrained YOLOv8 model for testing the API.

This script downloads YOLOv8n (nano - smallest/fastest) and saves it as weights.pt
so you can test the API without training your own model.

Note: This is a general object detection model trained on COCO dataset,
not specifically trained on NFL helmets. For accurate helmet detection,
you should train a custom model on the NFL dataset.

Usage:
    python scripts/download_pretrained_yolo.py
"""
import os
import sys


def download_model(model_size='n'):
    """
    Download a pretrained YOLOv8 model.

    Args:
        model_size: Size variant (n=nano, s=small, m=medium, l=large, x=xlarge)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found")
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
        from ultralytics import YOLO

    model_name = f'yolov8{model_size}.pt'
    output_path = 'models/weights.pt'

    print(f"Downloading YOLOv8{model_size.upper()} model...")
    print(f"Model: {model_name}")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Download the model
    model = YOLO(model_name)

    # Save to models/weights.pt
    print(f"Saving to {output_path}...")
    model.save(output_path)

    print(f"\n✓ Success! Model saved to {output_path}")
    print(f"\nModel info:")
    print(f"  - Variant: YOLOv8{model_size.upper()}")
    print(f"  - Training: COCO dataset (general objects)")
    print(f"  - Classes: 80 object classes (person, car, etc.)")
    print(f"\nNote: This is NOT trained on NFL helmets.")
    print(f"      For testing purposes only.")
    print(f"\nYou can now run the Docker container:")
    print(f"  docker build -t helmet-contact-detection:latest .")
    print(f"  docker run -p 8000:8000 \\")
    print(f"    -v \"$(pwd)/models/weights.pt:/app/models/weights.pt:ro\" \\")
    print(f"    helmet-contact-detection:latest")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download pretrained YOLOv8 model for testing'
    )
    parser.add_argument(
        '--size',
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge). Default: n'
    )

    args = parser.parse_args()

    print("="*60)
    print("YOLOv8 Pretrained Model Downloader")
    print("="*60 + "\n")

    download_model(args.size)


if __name__ == '__main__':
    main()
