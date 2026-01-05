"""Configuration management with YAML and environment variable support."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Data-related configuration."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    crops_dir: str = "data/processed/crops"
    metadata_file: str = "data/processed/metadata.parquet"

    tiny_mode: bool = False
    tiny_videos: int = 5
    tiny_max_samples: int = 500

    impact_filter: Dict[str, Any] = Field(
        default_factory=lambda: {
            "require_impact": True,
            "min_confidence": 0.5,
            "min_visibility": 0.5,
        }
    )

    crop_size: int = 224
    temporal_frames: int = 1
    temporal_window: int = 2


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    architecture: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.3
    input_size: int = 224


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = "cosine"

    patience: int = 5
    min_delta: float = 0.001

    val_split: float = 0.2
    split_by_play: bool = True

    balance_classes: bool = True
    pos_weight: float = 2.0


class EvaluationConfig(BaseModel):
    """Evaluation settings."""

    metrics: List[str] = ["accuracy", "precision", "recall", "f1", "auc_pr"]
    threshold: float = 0.5
    save_predictions: bool = True


class InferenceConfig(BaseModel):
    """Inference pipeline configuration."""

    model_path: str = "models/best.pt"
    onnx_path: str = "models/model.onnx"
    use_onnx: bool = False

    detector_model: str = "yolov8n.pt"
    detector_conf: float = 0.3
    detector_iou: float = 0.5
    helmet_class_id: int = 0

    sample_rate: int = 5
    min_score: float = 0.5

    annotate_video: bool = True
    annotation_color: List[int] = [0, 0, 255]
    annotation_thickness: int = 2


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size: int = 524288000
    allowed_extensions: List[str] = [".mp4", ".avi", ".mov"]
    temp_dir: str = "/tmp/impact_detector"


class PathsConfig(BaseModel):
    """Path configuration."""

    models_dir: str = "models"
    checkpoints_dir: str = "models/checkpoints"
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"


class Config(BaseSettings):
    """Main configuration class."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to YAML config file. If None, uses base.yaml

    Returns:
        Loaded configuration object
    """
    if config_path is None:
        config_path = "configs/base.yaml"

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return Config()

    # Load YAML
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Create config from dict
    config = Config(**config_dict)

    # Environment variables will override via pydantic-settings
    # Format: SECTION__KEY, e.g., DATA__TINY_MODE=true

    return config


def merge_configs(base_path: str, override_path: str) -> Config:
    """Merge two config files, with override taking precedence.

    Args:
        base_path: Path to base configuration
        override_path: Path to override configuration

    Returns:
        Merged configuration
    """
    # Load base
    with open(base_path) as f:
        base_dict = yaml.safe_load(f)

    # Load override
    with open(override_path) as f:
        override_dict = yaml.safe_load(f)

    # Deep merge
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(base_dict, override_dict)
    return Config(**merged)
