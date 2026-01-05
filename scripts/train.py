"""Training script for helmet impact detection model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.impact_detector.config import load_config, merge_configs
from src.impact_detector.dataset import (
    ImpactDataset,
    create_balanced_sampler,
    create_train_val_split,
    get_transforms,
)
from src.impact_detector.model import create_model


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # Precision-Recall AUC
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        # Simple AUC approximation
        metrics["auc_pr"] = np.trapz(precision, recall)
    else:
        metrics["auc_pr"] = 0.0

    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        (loss, metrics, predictions)
    """
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class

    # Compute metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics, (all_labels, all_preds, all_probs)


def train(config):
    """Main training function.

    Args:
        config: Configuration object
    """
    print("=" * 60)
    print("Training Helmet Impact Detection Model")
    print("=" * 60)

    # Set seed
    set_seed(config.training.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load metadata
    print("\nLoading metadata...")
    metadata_path = Path(config.data.metadata_file)

    if not metadata_path.exists():
        print(f"✗ Metadata file not found: {metadata_path}")
        print("Please run build_crops_cache.py first")
        return False

    metadata_df = pd.read_parquet(metadata_path)
    print(f"Loaded {len(metadata_df)} samples")

    # Train/val split
    print("\nCreating train/val split...")
    train_df, val_df = create_train_val_split(
        metadata_df,
        val_split=config.training.val_split,
        split_by_play=config.training.split_by_play,
        seed=config.training.seed,
    )

    # Create datasets
    print("\nCreating datasets...")
    train_transform = get_transforms(train=True, input_size=config.model.input_size)
    val_transform = get_transforms(train=False, input_size=config.model.input_size)

    train_dataset = ImpactDataset(
        train_df,
        transform=train_transform,
        temporal_frames=config.data.temporal_frames,
    )

    val_dataset = ImpactDataset(
        val_df,
        transform=val_transform,
        temporal_frames=config.data.temporal_frames,
    )

    # Create dataloaders
    print("\nCreating dataloaders...")

    if config.training.balance_classes:
        train_sampler = create_balanced_sampler(
            train_dataset,
            pos_weight=config.training.pos_weight,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        architecture=config.model.architecture,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        temporal_frames=config.data.temporal_frames,
    )
    model = model.to(device)

    print(f"Model: {config.model.architecture}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Scheduler
    if config.training.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    else:
        scheduler = None

    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0
    patience_counter = 0

    models_dir = Path(config.paths.models_dir)
    models_dir.mkdir(exist_ok=True)

    checkpoints_dir = Path(config.paths.checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True)

    for epoch in range(config.training.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.training.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device)

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Learning rate
        if scheduler:
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "model_config": {
                "architecture": config.model.architecture,
                "pretrained": False,  # Don't reload pretrained weights
                "num_classes": config.model.num_classes,
                "dropout": config.model.dropout,
                "temporal_frames": config.data.temporal_frames,
            },
        }

        # Save latest
        torch.save(checkpoint, checkpoints_dir / "latest.pt")

        # Save best
        current_f1 = val_metrics["f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(checkpoint, models_dir / "best.pt")
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.training.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best model saved to: {models_dir / 'best.pt'}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Train impact detection model")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--override-config",
        help="Optional override config (e.g., configs/tiny.yaml)",
    )

    args = parser.parse_args()

    # Load config
    if args.override_config:
        config = merge_configs(args.config, args.override_config)
    else:
        config = load_config(args.config)

    # Train
    success = train(config)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
