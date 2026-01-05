"""PyTorch datasets for NFL helmet impact detection."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImpactDataset(Dataset):
    """Dataset for helmet impact detection."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        temporal_frames: int = 1,
    ):
        """Initialize dataset.

        Args:
            metadata_df: DataFrame with crop_path and label columns
            transform: Optional image transforms
            temporal_frames: Number of temporal frames (1 for single frame)
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.temporal_frames = temporal_frames

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            (image, label) tuple
        """
        row = self.metadata.iloc[idx]
        crop_path = Path(row["crop_path"])
        label = int(row["label"])

        # Load crop
        if crop_path.suffix == ".npy":
            # Temporal stack
            crop = np.load(crop_path)  # (T, H, W, C)
        else:
            # Single frame JPEG
            crop = cv2.imread(str(crop_path))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            if self.temporal_frames > 1:
                # Apply transform to each frame
                transformed_frames = []
                for frame in crop:
                    frame = self.transform(frame)
                    transformed_frames.append(frame)
                image = torch.stack(transformed_frames)  # (T, C, H, W)
            else:
                image = self.transform(crop)  # (C, H, W)
        else:
            # Convert to tensor manually
            if self.temporal_frames > 1:
                image = torch.from_numpy(crop).permute(0, 3, 1, 2).float() / 255.0
            else:
                image = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

        return image, label


def get_transforms(train: bool = True, input_size: int = 224) -> transforms.Compose:
    """Get image transforms.

    Args:
        train: Whether this is for training (includes augmentation)
        input_size: Input image size

    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_train_val_split(
    metadata_df: pd.DataFrame,
    val_split: float = 0.2,
    split_by_play: bool = True,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation split.

    Args:
        metadata_df: Full metadata DataFrame
        val_split: Fraction for validation
        split_by_play: Whether to split by (game_key, play_id) to avoid leakage
        seed: Random seed

    Returns:
        (train_df, val_df) tuple
    """
    np.random.seed(seed)

    if split_by_play and "game_key" in metadata_df.columns and "play_id" in metadata_df.columns:
        # Split by unique plays
        plays = metadata_df[["game_key", "play_id"]].drop_duplicates()
        n_val = int(len(plays) * val_split)

        val_plays = plays.sample(n=n_val, random_state=seed)

        # Split metadata
        val_mask = metadata_df.set_index(["game_key", "play_id"]).index.isin(
            val_plays.set_index(["game_key", "play_id"]).index
        )

        val_df = metadata_df[val_mask]
        train_df = metadata_df[~val_mask]

        print(f"Split by play: {len(train_df)} train, {len(val_df)} val")
        print(f"  Unique plays: {len(plays) - n_val} train, {n_val} val")

    else:
        # Random split
        n_val = int(len(metadata_df) * val_split)
        indices = np.random.permutation(len(metadata_df))

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        val_df = metadata_df.iloc[val_indices]
        train_df = metadata_df.iloc[train_indices]

        print(f"Random split: {len(train_df)} train, {len(val_df)} val")

    return train_df, val_df


def create_balanced_sampler(
    dataset: ImpactDataset,
    pos_weight: float = 1.0,
) -> torch.utils.data.WeightedRandomSampler:
    """Create a balanced sampler for imbalanced datasets.

    Args:
        dataset: Dataset instance
        pos_weight: Weight multiplier for positive class

    Returns:
        WeightedRandomSampler
    """
    labels = dataset.metadata["label"].values

    # Count classes
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()

    print(f"Class distribution: {neg_count} negative, {pos_count} positive")

    # Compute weights
    weights = np.ones(len(labels))
    weights[labels == 1] = pos_weight

    # Normalize
    weights = weights / weights.sum() * len(weights)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler
