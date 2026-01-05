"""Export trained model to ONNX format."""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.impact_detector.config import load_config
from src.impact_detector.model import load_checkpoint


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: int = 224,
    temporal_frames: int = 1,
    opset_version: int = 14,
):
    """Export PyTorch model to ONNX.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_size: Input image size
        temporal_frames: Number of temporal frames
        opset_version: ONNX opset version
    """
    model.eval()

    # Create dummy input
    if temporal_frames > 1:
        dummy_input = torch.randn(1, temporal_frames, 3, input_size, input_size)
    else:
        dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export
    print(f"Exporting model to {output_path}...")
    print(f"  Input shape: {dummy_input.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"✓ Exported to {output_path}")


def verify_onnx(onnx_path: str, input_size: int = 224, temporal_frames: int = 1):
    """Verify ONNX model.

    Args:
        onnx_path: Path to ONNX model
        input_size: Input image size
        temporal_frames: Number of temporal frames
    """
    print(f"\nVerifying ONNX model...")

    # Check model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("✓ ONNX model is valid")

    # Test inference
    print("\nTesting inference...")
    session = ort.InferenceSession(onnx_path)

    # Create dummy input
    if temporal_frames > 1:
        dummy_input = np.random.randn(1, temporal_frames, 3, input_size, input_size).astype(
            np.float32
        )
    else:
        dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    # Run inference
    outputs = session.run(None, {"input": dummy_input})

    print(f"✓ Inference successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {outputs[0].shape}")

    # Print model info
    print("\nModel Info:")
    print(f"  Inputs: {session.get_inputs()[0].name}, shape: {session.get_inputs()[0].shape}")
    print(f"  Outputs: {session.get_outputs()[0].name}, shape: {session.get_outputs()[0].shape}")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
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
        "--output",
        default="models/model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported model",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device("cpu")  # Export on CPU
    model = load_checkpoint(args.checkpoint, device)

    # Export
    export_to_onnx(
        model,
        args.output,
        input_size=config.model.input_size,
        temporal_frames=config.data.temporal_frames,
    )

    # Verify
    if args.verify:
        verify_onnx(
            args.output,
            input_size=config.model.input_size,
            temporal_frames=config.data.temporal_frames,
        )

    print("\n✓ Export complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
