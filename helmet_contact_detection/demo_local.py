#!/usr/bin/env python3
"""
Local demo script for NFL Helmet Contact Detection API.

This script demonstrates the API functionality without requiring Docker.
It downloads a pretrained YOLO model and starts the API server.

Usage:
    python demo_local.py
"""
import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed."""
    required = ['fastapi', 'uvicorn', 'ultralytics', 'opencv-python-headless', 'pillow']
    missing = []

    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("\nInstalling dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
        print("✓ Dependencies installed")


def download_pretrained_model():
    """Download a pretrained YOLOv8 model for demonstration."""
    from ultralytics import YOLO

    model_path = "models/weights.pt"

    if os.path.exists(model_path):
        print(f"✓ Model already exists at {model_path}")
        return model_path

    print("Downloading pretrained YOLOv8n model for demonstration...")
    print("(Note: This is a general object detection model, not trained on helmets)")

    # Download YOLOv8n - smallest and fastest model
    model = YOLO('yolov8n.pt')

    # Save it to the models directory
    os.makedirs('models', exist_ok=True)
    model.save(model_path)

    print(f"✓ Model saved to {model_path}")
    return model_path


def start_server():
    """Start the FastAPI server."""
    print("\n" + "="*60)
    print("Starting NFL Helmet Contact Detection API")
    print("="*60)
    print("\nAPI Endpoints:")
    print("  • Root:           http://localhost:8000/")
    print("  • Health:         http://localhost:8000/health")
    print("  • Docs:           http://localhost:8000/docs")
    print("  • Predict Frame:  http://localhost:8000/predict_frame")
    print("  • Predict Clip:   http://localhost:8000/predict_clip")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")

    # Start uvicorn server
    subprocess.check_call([
        sys.executable, '-m', 'uvicorn',
        'app.main:app',
        '--host', '0.0.0.0',
        '--port', '8000',
        '--reload'
    ])


def main():
    """Main demo function."""
    print("NFL Helmet Contact Detection - Local Demo")
    print("="*60 + "\n")

    # Step 1: Check dependencies
    print("Step 1: Checking dependencies...")
    check_dependencies()

    # Step 2: Download model
    print("\nStep 2: Setting up model...")
    download_pretrained_model()

    # Step 3: Start server
    print("\nStep 3: Starting server...")
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        print("Demo complete!")


if __name__ == "__main__":
    main()
