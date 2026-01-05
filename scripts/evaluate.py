"""Evaluate trained model and generate metrics."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.impact_detector.config import load_config
from src.impact_detector.dataset import ImpactDataset, create_train_val_split, get_transforms
from src.impact_detector.model import load_checkpoint


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions."""
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved confusion matrix to {output_path}")


def plot_pr_curve(y_true, y_probs, output_path):
    """Plot precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved PR curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint, device)

    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_parquet(config.data.metadata_file)

    # Use validation split
    _, val_df = create_train_val_split(
        metadata_df,
        val_split=config.training.val_split,
        split_by_play=config.training.split_by_play,
        seed=config.training.seed,
    )

    print(f"Evaluating on {len(val_df)} validation samples")

    # Create dataset
    val_transform = get_transforms(train=False, input_size=config.model.input_size)
    val_dataset = ImpactDataset(
        val_df,
        transform=val_transform,
        temporal_frames=config.data.temporal_frames,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    # Evaluate
    print("\nEvaluating...")
    y_true, y_pred, y_probs = evaluate_model(model, val_loader, device)

    # Print classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["No Impact", "Impact"]))

    # Save predictions
    if config.evaluation.save_predictions:
        predictions_df = pd.DataFrame({
            "true_label": y_true,
            "predicted_label": y_pred,
            "impact_probability": y_probs,
        })
        predictions_path = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\nSaved predictions to {predictions_path}")

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    # Plot PR curve
    plot_pr_curve(y_true, y_probs, output_dir / "pr_curve.png")

    print("\n✓ Evaluation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
