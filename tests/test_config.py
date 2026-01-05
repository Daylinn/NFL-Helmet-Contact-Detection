"""Tests for configuration module."""

from pathlib import Path

import pytest

from src.impact_detector.config import Config, load_config, merge_configs


def test_default_config():
    """Test default configuration."""
    config = Config()

    assert config.data.raw_dir == "data/raw"
    assert config.model.architecture == "resnet18"
    assert config.training.batch_size == 32


def test_load_config():
    """Test loading config from YAML."""
    config = load_config("configs/base.yaml")

    assert config is not None
    assert config.data.crop_size == 224
    assert config.model.num_classes == 2


def test_load_missing_config():
    """Test loading non-existent config falls back to defaults."""
    config = load_config("configs/nonexistent.yaml")

    assert config is not None
    # Should use default values
    assert config.data.raw_dir == "data/raw"


def test_tiny_config():
    """Test tiny mode configuration."""
    config = load_config("configs/tiny.yaml")

    # Should have base + tiny overrides
    assert config.data.tiny_mode is True
    assert config.data.tiny_videos == 3
    assert config.training.epochs == 5
