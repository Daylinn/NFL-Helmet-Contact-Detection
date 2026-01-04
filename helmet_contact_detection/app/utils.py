"""
Utility functions for image and video processing.
"""
import io
import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes to numpy array (BGR format for OpenCV).

    Args:
        image_bytes: Raw image bytes

    Returns:
        np.ndarray: Image in BGR format (H, W, C)

    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB then to BGR for OpenCV
        image_rgb = np.array(image.convert('RGB'))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def load_video_from_bytes(video_bytes: bytes) -> List[np.ndarray]:
    """
    Load video from bytes and extract all frames.

    Args:
        video_bytes: Raw video bytes

    Returns:
        List[np.ndarray]: List of frames in BGR format

    Raises:
        ValueError: If video cannot be decoded
    """
    try:
        # Write bytes to temporary in-memory file
        temp_path = "/tmp/temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)

        # Open video
        cap = cv2.VideoCapture(temp_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError("No frames could be extracted from video")

        return frames

    except Exception as e:
        raise ValueError(f"Failed to decode video: {str(e)}")


def sample_frames(frames: List[np.ndarray], max_frames: int = 30) -> Tuple[List[np.ndarray], List[int]]:
    """
    Sample frames uniformly from a list of frames.

    Args:
        frames: List of all frames
        max_frames: Maximum number of frames to sample

    Returns:
        Tuple of (sampled_frames, frame_indices)
    """
    total_frames = len(frames)

    if total_frames <= max_frames:
        return frames, list(range(total_frames))

    # Sample uniformly
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    sampled = [frames[i] for i in indices]

    return sampled, indices.tolist()


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) format
        box2: (x1, y1, x2, y2) format

    Returns:
        float: IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculate center point of a bounding box.

    Args:
        box: (x1, y1, x2, y2) format

    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def calculate_euclidean_distance(point1: Tuple[float, float],
                                  point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: (x, y)
        point2: (x, y)

    Returns:
        float: Distance
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
