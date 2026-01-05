"""Tests for model module."""

import torch

from src.impact_detector.model import ImpactClassifier, create_model


def test_create_model():
    """Test model creation."""
    model = create_model(
        architecture="resnet18",
        pretrained=False,  # Faster for testing
        num_classes=2,
        dropout=0.3,
        temporal_frames=1,
    )

    assert isinstance(model, ImpactClassifier)
    assert model.num_classes == 2
    assert model.temporal_frames == 1


def test_model_forward_single_frame():
    """Test forward pass with single frame."""
    model = create_model(
        architecture="resnet18",
        pretrained=False,
        num_classes=2,
        temporal_frames=1,
    )

    # Create dummy input
    x = torch.randn(4, 3, 224, 224)  # (B, C, H, W)

    # Forward pass
    output = model(x)

    assert output.shape == (4, 2)  # (B, num_classes)


def test_model_forward_temporal():
    """Test forward pass with temporal frames."""
    model = create_model(
        architecture="resnet18",
        pretrained=False,
        num_classes=2,
        temporal_frames=5,
    )

    # Create dummy input
    x = torch.randn(4, 5, 3, 224, 224)  # (B, T, C, H, W)

    # Forward pass
    output = model(x)

    assert output.shape == (4, 2)  # (B, num_classes)


def test_model_predict():
    """Test prediction methods."""
    model = create_model(
        architecture="resnet18",
        pretrained=False,
        num_classes=2,
    )

    x = torch.randn(2, 3, 224, 224)

    # Test predict_proba
    probs = model.predict_proba(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    # Test predict
    preds = model.predict(x)
    assert preds.shape == (2,)
    assert all(p in [0, 1] for p in preds.tolist())
