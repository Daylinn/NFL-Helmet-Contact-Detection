"""Model architecture for helmet impact detection."""

from typing import Optional

import timm
import torch
import torch.nn as nn


class ImpactClassifier(nn.Module):
    """Helmet impact classifier based on timm backbone."""

    def __init__(
        self,
        architecture: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        temporal_frames: int = 1,
    ):
        """Initialize model.

        Args:
            architecture: timm model name
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes
            dropout: Dropout rate
            temporal_frames: Number of temporal frames (1 for single frame)
        """
        super().__init__()

        self.temporal_frames = temporal_frames
        self.num_classes = num_classes

        # Create backbone
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Temporal aggregation if needed
        if temporal_frames > 1:
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            # Note: For simplicity, we'll process each frame independently
            # and average the features. More sophisticated approaches could use
            # 3D convolutions or transformers.

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape:
               - (B, C, H, W) for single frame
               - (B, T, C, H, W) for temporal

        Returns:
            Logits of shape (B, num_classes)
        """
        if self.temporal_frames > 1:
            # Process temporal frames
            B, T, C, H, W = x.shape

            # Reshape to (B*T, C, H, W)
            x = x.view(B * T, C, H, W)

            # Extract features
            features = self.backbone(x)  # (B*T, D)

            # Reshape back to (B, T, D)
            D = features.shape[1]
            features = features.view(B, T, D)

            # Temporal pooling (average across time)
            features = features.mean(dim=1)  # (B, D)
        else:
            # Single frame
            features = self.backbone(x)

        # Classification
        logits = self.classifier(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            x: Input tensor

        Returns:
            Probabilities of shape (B, num_classes)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions.

        Args:
            x: Input tensor

        Returns:
            Predicted classes of shape (B,)
        """
        probs = self.predict_proba(x)
        preds = torch.argmax(probs, dim=1)
        return preds


def create_model(
    architecture: str = "resnet18",
    pretrained: bool = True,
    num_classes: int = 2,
    dropout: float = 0.3,
    temporal_frames: int = 1,
) -> ImpactClassifier:
    """Create model instance.

    Args:
        architecture: timm model name
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate
        temporal_frames: Number of temporal frames

    Returns:
        Model instance
    """
    model = ImpactClassifier(
        architecture=architecture,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        temporal_frames=temporal_frames,
    )

    return model


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> ImpactClassifier:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get("model_config", {})

    # Create model
    model = create_model(**model_config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model
