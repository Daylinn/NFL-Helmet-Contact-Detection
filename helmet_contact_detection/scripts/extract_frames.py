#!/usr/bin/env python3
"""
Utility script to extract frames from NFL game videos for testing.

Usage:
    python scripts/extract_frames.py <video_path> <output_dir> [--fps 1] [--max-frames 100]
"""
import argparse
import cv2
from pathlib import Path
import sys


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int = None
) -> int:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
        max_frames: Maximum number of frames to extract (optional)

    Returns:
        Number of frames extracted

    Raises:
        FileNotFoundError: If video file not found
        ValueError: If video cannot be opened
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path.name}")
    print(f"FPS: {video_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting at {fps} fps")

    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps < video_fps else 1

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = output_dir / f"frame_{extracted_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            extracted_count += 1

            if extracted_count % 10 == 0:
                print(f"Extracted {extracted_count} frames...", end='\r')

            if max_frames and extracted_count >= max_frames:
                break

        frame_count += 1

    cap.release()

    print(f"\nExtracted {extracted_count} frames to {output_dir}")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video for helmet detection testing"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: no limit)"
    )

    args = parser.parse_args()

    try:
        extract_frames(
            video_path=args.video_path,
            output_dir=args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames
        )
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
