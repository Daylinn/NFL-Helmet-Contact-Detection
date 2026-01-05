"""Data utilities for loading and parsing NFL impact detection data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def discover_data_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Auto-discover data files in the raw data directory.

    Args:
        data_dir: Path to raw data directory

    Returns:
        Dictionary with 'labels', 'tracking', and 'videos' keys
    """
    discovered = {"labels": [], "tracking": [], "videos": []}

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        name_lower = file_path.name.lower()

        if file_path.suffix == ".csv":
            if "label" in name_lower or "video" in name_lower:
                discovered["labels"].append(file_path)
            elif "track" in name_lower or "player" in name_lower:
                discovered["tracking"].append(file_path)
        elif file_path.suffix in [".mp4", ".avi", ".mov"]:
            discovered["videos"].append(file_path)

    return discovered


def load_video_labels(
    labels_path: Path,
    impact_filter: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Load and filter video labels.

    Args:
        labels_path: Path to labels CSV
        impact_filter: Filter configuration for definitive impacts

    Returns:
        Filtered DataFrame with video labels
    """
    df = pd.read_csv(labels_path)

    print(f"Loaded {len(df)} total labels from {labels_path.name}")

    if impact_filter is None:
        return df

    # Apply filters
    original_len = len(df)

    # Filter for impacts
    if impact_filter.get("require_impact", True):
        if "impact" in df.columns:
            df = df[df["impact"] == 1]
            print(f"  Filtered to {len(df)} impacts (impact==1)")

    # Confidence filter
    min_conf = impact_filter.get("min_confidence", 0.0)
    if min_conf > 0 and "confidence" in df.columns:
        df = df[df["confidence"] >= min_conf]
        print(f"  Filtered to {len(df)} with confidence>={min_conf}")

    # Visibility filter
    min_vis = impact_filter.get("min_visibility", 0.0)
    if min_vis > 0 and "visibility" in df.columns:
        df = df[df["visibility"] >= min_vis]
        print(f"  Filtered to {len(df)} with visibility>={min_vis}")

    print(f"  Final: {len(df)}/{original_len} labels after filtering")

    return df


def parse_bbox(row: pd.Series) -> Tuple[int, int, int, int]:
    """Parse bounding box from label row.

    Args:
        row: DataFrame row with bbox info

    Returns:
        (x1, y1, x2, y2) bounding box coordinates
    """
    # Check different possible column names
    if "left" in row and "width" in row:
        x1 = int(row["left"])
        y1 = int(row["top"])
        w = int(row["width"])
        h = int(row["height"])
        return x1, y1, x1 + w, y1 + h

    elif "x1" in row:
        return int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

    else:
        raise ValueError(f"Cannot parse bbox from columns: {row.index.tolist()}")


def extract_crop(
    video_path: Path,
    frame_idx: int,
    bbox: Tuple[int, int, int, int],
    target_size: int = 224,
) -> Optional[np.ndarray]:
    """Extract a cropped region from a video frame.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        bbox: (x1, y1, x2, y2) bounding box
        target_size: Resize crop to this size

    Returns:
        Cropped image array (RGB) or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))

    try:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Resize
        crop = cv2.resize(crop, (target_size, target_size))

        return crop

    finally:
        cap.release()


def extract_temporal_crop(
    video_path: Path,
    center_frame: int,
    bbox: Tuple[int, int, int, int],
    temporal_window: int = 2,
    target_size: int = 224,
) -> Optional[np.ndarray]:
    """Extract a temporal stack of crops around a center frame.

    Args:
        video_path: Path to video file
        center_frame: Center frame index
        bbox: (x1, y1, x2, y2) bounding box
        temporal_window: Number of frames before/after center
        target_size: Resize crops to this size

    Returns:
        Stacked crops array of shape (T, H, W, C) or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, center_frame - temporal_window)
        end_frame = min(total_frames, center_frame + temporal_window + 1)

        crops = []

        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (target_size, target_size))
            crops.append(crop)

        if not crops:
            return None

        # Stack along time dimension
        return np.stack(crops, axis=0)

    finally:
        cap.release()


def create_video_index(data_dir: Path) -> Dict[str, Path]:
    """Create a mapping from video identifiers to file paths.

    Args:
        data_dir: Raw data directory

    Returns:
        Dictionary mapping video names/IDs to paths
    """
    video_index = {}

    for video_path in data_dir.rglob("*.mp4"):
        # Try different naming conventions
        video_name = video_path.stem
        video_index[video_name] = video_path

        # Also index without extensions or prefixes
        clean_name = video_name.split("_")[0] if "_" in video_name else video_name
        video_index[clean_name] = video_path

    return video_index


def build_sample_metadata(
    labels_df: pd.DataFrame,
    video_index: Dict[str, Path],
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Build metadata for all training samples.

    Args:
        labels_df: Filtered labels DataFrame
        video_index: Mapping from video IDs to paths
        max_samples: Maximum samples to include (for tiny mode)

    Returns:
        Metadata DataFrame with columns: video_path, frame, bbox, label, etc.
    """
    samples = []

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Building metadata"):
        # Find video path
        video_id = None
        video_path = None

        # Try different column names for video identifier
        for col in ["video", "videoName", "video_name", "gameKey"]:
            if col in row and pd.notna(row[col]):
                video_id = str(row[col])
                if video_id in video_index:
                    video_path = video_index[video_id]
                    break

        if video_path is None:
            continue  # Skip if video not found

        # Parse frame
        frame = int(row.get("frame", 0))

        # Parse bbox
        try:
            bbox = parse_bbox(row)
        except (KeyError, ValueError):
            continue  # Skip if bbox invalid

        # Build sample
        sample = {
            "video_path": str(video_path),
            "frame": frame,
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "label": 1,  # Impact (filtered data)
        }

        # Add optional fields
        if "gameKey" in row:
            sample["game_key"] = row["gameKey"]
        if "playID" in row:
            sample["play_id"] = row["playID"]
        if "confidence" in row:
            sample["confidence"] = row["confidence"]

        samples.append(sample)

        if max_samples and len(samples) >= max_samples:
            break

    return pd.DataFrame(samples)


def validate_sample(sample: Dict[str, Any], crop_size: int = 224) -> bool:
    """Validate that a sample can be successfully processed.

    Args:
        sample: Sample dictionary with video_path, frame, bbox
        crop_size: Expected crop size

    Returns:
        True if sample is valid, False otherwise
    """
    try:
        video_path = Path(sample["video_path"])
        if not video_path.exists():
            return False

        bbox = (sample["x1"], sample["y1"], sample["x2"], sample["y2"])
        crop = extract_crop(video_path, sample["frame"], bbox, crop_size)

        return crop is not None and crop.shape == (crop_size, crop_size, 3)

    except Exception:
        return False
