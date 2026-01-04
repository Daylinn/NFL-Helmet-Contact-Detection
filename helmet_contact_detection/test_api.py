#!/usr/bin/env python3
"""
Simple test script for the NFL Helmet Contact Detection API.

Usage:
    python test_api.py [--url http://localhost:8000]
"""
import argparse
import sys
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np


def create_test_image(width: int = 1280, height: int = 720) -> bytes:
    """
    Create a simple test image with colored rectangles.

    Args:
        width: Image width
        height: Image height

    Returns:
        Image bytes in JPEG format
    """
    # Create blank image
    img = Image.new('RGB', (width, height), color='green')
    draw = ImageDraw.Draw(img)

    # Draw some rectangles to simulate helmets
    rectangles = [
        (200, 300, 280, 380, 'blue'),
        (500, 350, 580, 430, 'red'),
        (800, 320, 880, 400, 'yellow'),
    ]

    for x1, y1, x2, y2, color in rectangles:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='white', width=3)

    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def test_health(base_url: str) -> bool:
    """
    Test the health endpoint.

    Args:
        base_url: Base URL of the API

    Returns:
        True if healthy, False otherwise
    """
    print(f"Testing health endpoint: {base_url}/health")

    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()

        data = response.json()
        print(f"✓ Health check passed")
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Version: {data['version']}")

        return data['status'] == 'healthy'

    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_predict_frame(base_url: str) -> bool:
    """
    Test the predict_frame endpoint.

    Args:
        base_url: Base URL of the API

    Returns:
        True if successful, False otherwise
    """
    print(f"\nTesting predict_frame endpoint: {base_url}/predict_frame")

    try:
        # Create test image
        image_bytes = create_test_image()

        # Send request
        files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(
            f"{base_url}/predict_frame",
            files=files,
            timeout=30
        )

        if response.status_code == 503:
            print("✗ Model not loaded (expected if weights.pt is missing)")
            print("  To fix: Place trained YOLO weights at models/weights.pt")
            return False

        response.raise_for_status()

        data = response.json()
        print(f"✓ Prediction successful")
        print(f"  Helmets detected: {len(data['helmets'])}")
        print(f"  Potential contacts: {len(data['contacts'])}")
        print(f"  Frame has contact: {data['frame_has_contact']}")
        print(f"  Inference time: {data['inference_time_ms']:.2f}ms")

        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            print("✗ Model not loaded (expected if weights.pt is missing)")
            print("  To fix: Place trained YOLO weights at models/weights.pt")
        else:
            print(f"✗ Prediction failed: {e}")
        return False

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test the NFL Helmet Contact Detection API"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NFL Helmet Contact Detection API Test")
    print("=" * 60)

    # Test health endpoint
    health_ok = test_health(args.url)

    if not health_ok:
        print("\n✗ API is not healthy. Please check the service.")
        return 1

    # Test predict_frame endpoint
    predict_ok = test_predict_frame(args.url)

    print("\n" + "=" * 60)
    if health_ok and predict_ok:
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
