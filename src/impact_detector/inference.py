"""Inference pipeline for detecting helmet impacts in video."""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


class HelmetDetector:
    """Helmet detector using YOLOv8."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.3, iou: float = 0.5):
        """Initialize detector.

        Args:
            model_path: Path to YOLO model
            conf: Confidence threshold
            iou: IOU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect helmets/persons in frame.

        Args:
            frame: Image array (RGB)

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)

        bboxes = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Filter for person class (class 0) as proxy for helmet
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))

        return bboxes


class ImpactClassifierInference:
    """Impact classifier for inference."""

    def __init__(
        self,
        model_path: str,
        use_onnx: bool = False,
        device: Optional[torch.device] = None,
        input_size: int = 224,
    ):
        """Initialize classifier.

        Args:
            model_path: Path to model checkpoint or ONNX file
            use_onnx: Whether to use ONNX runtime
            device: Device for PyTorch (ignored for ONNX)
            input_size: Input image size
        """
        self.use_onnx = use_onnx
        self.input_size = input_size

        if use_onnx:
            self.session = ort.InferenceSession(model_path)
        else:
            from src.impact_detector.model import load_checkpoint

            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.device = device
            self.model = load_checkpoint(model_path, device)

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess crop for inference.

        Args:
            crop: Image crop (RGB)

        Returns:
            Preprocessed tensor
        """
        pil_image = Image.fromarray(crop)
        tensor = self.transform(pil_image)

        if self.use_onnx:
            return tensor.numpy()
        else:
            return tensor

    def predict(self, crop: np.ndarray) -> Tuple[int, float]:
        """Predict impact for a crop.

        Args:
            crop: Image crop (RGB)

        Returns:
            (predicted_class, confidence) tuple
        """
        tensor = self.preprocess(crop)

        if self.use_onnx:
            # ONNX inference
            tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
            outputs = self.session.run(None, {"input": tensor})
            logits = outputs[0][0]

            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
        else:
            # PyTorch inference
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)

            pred_class = int(torch.argmax(probs, dim=1).cpu().numpy()[0])
            confidence = float(probs[0, pred_class].cpu().numpy())

        return pred_class, confidence


class ImpactDetectionPipeline:
    """End-to-end pipeline for detecting impacts in video."""

    def __init__(
        self,
        classifier_path: str,
        detector_path: str = "yolov8n.pt",
        use_onnx: bool = False,
        detector_conf: float = 0.3,
        detector_iou: float = 0.5,
        sample_rate: int = 5,
        min_score: float = 0.5,
        input_size: int = 224,
    ):
        """Initialize pipeline.

        Args:
            classifier_path: Path to impact classifier model
            detector_path: Path to helmet detector model
            use_onnx: Whether to use ONNX for classifier
            detector_conf: Detector confidence threshold
            detector_iou: Detector IOU threshold
            sample_rate: Process every N frames
            min_score: Minimum impact score to report
            input_size: Classifier input size
        """
        self.detector = HelmetDetector(detector_path, detector_conf, detector_iou)
        self.classifier = ImpactClassifierInference(classifier_path, use_onnx, input_size=input_size)
        self.sample_rate = sample_rate
        self.min_score = min_score

    def process_video(
        self,
        video_path: str,
        annotate: bool = False,
        output_path: Optional[str] = None,
    ) -> List[dict]:
        """Process video and detect impacts.

        Args:
            video_path: Path to input video
            annotate: Whether to create annotated video
            output_path: Path for annotated output (if annotate=True)

        Returns:
            List of detected impacts with metadata
        """
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        detections = []

        # Video writer for annotation
        writer = None
        if annotate:
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process only sampled frames
                if frame_idx % self.sample_rate == 0:
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect helmets
                    bboxes = self.detector.detect(frame_rgb)

                    # Classify each detection
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox

                        # Extract crop
                        crop = frame_rgb[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        # Predict impact
                        pred_class, confidence = self.classifier.predict(crop)

                        # Check if impact detected
                        if pred_class == 1 and confidence >= self.min_score:
                            detection = {
                                "frame": frame_idx,
                                "time_sec": frame_idx / fps,
                                "bbox": [x1, y1, x2, y2],
                                "score": confidence,
                            }
                            detections.append(detection)

                            # Annotate frame
                            if annotate:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                label = f"Impact: {confidence:.2f}"
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    2,
                                )

                # Write frame
                if writer:
                    writer.write(frame)

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()

        print(f"Processed {frame_idx} frames, found {len(detections)} impacts")

        if annotate and output_path:
            print(f"Annotated video saved to {output_path}")

        return detections, output_path if annotate else None


def run_inference(
    video_path: str,
    config,
    annotate: bool = False,
    output_path: Optional[str] = None,
) -> Tuple[List[dict], Optional[str]]:
    """Run inference on a video file.

    Args:
        video_path: Path to input video
        config: Configuration object
        annotate: Whether to create annotated video
        output_path: Path for annotated output

    Returns:
        (detections, annotated_video_path) tuple
    """
    # Create pipeline
    model_path = config.inference.onnx_path if config.inference.use_onnx else config.inference.model_path

    pipeline = ImpactDetectionPipeline(
        classifier_path=model_path,
        detector_path=config.inference.detector_model,
        use_onnx=config.inference.use_onnx,
        detector_conf=config.inference.detector_conf,
        detector_iou=config.inference.detector_iou,
        sample_rate=config.inference.sample_rate,
        min_score=config.inference.min_score,
        input_size=config.model.input_size,
    )

    # Process video
    detections, output = pipeline.process_video(video_path, annotate, output_path)

    return detections, output
