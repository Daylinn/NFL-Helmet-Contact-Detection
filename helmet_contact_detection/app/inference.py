"""
Model inference logic for helmet detection and contact prediction.
"""
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from app.schemas import (
    HelmetDetection,
    ContactPrediction,
    BoundingBox,
    FramePredictionResponse,
    ClipPredictionResponse
)
from app.utils import (
    calculate_iou,
    calculate_box_center,
    calculate_euclidean_distance,
    sample_frames
)

logger = logging.getLogger(__name__)


class HelmetContactDetector:
    """
    Helmet detection and contact prediction model.

    Uses YOLO for helmet detection and geometric heuristics for contact prediction.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO weights file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._weights_loaded = False

        # Contact detection thresholds
        self.iou_threshold = 0.1  # Minimum IoU to consider contact
        self.distance_threshold = 100  # Maximum center distance (pixels) for contact

    def load_model(self) -> None:
        """
        Load the YOLO model from weights file.

        Raises:
            FileNotFoundError: If model weights not found
            RuntimeError: If model fails to load
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. "
                "Please place trained YOLO weights at this path or see "
                "scripts/download_kaggle_instructions.md for dataset info."
            )

        try:
            # Import here to avoid loading during initialization
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(str(self.model_path))
            self._weights_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    @property
    def weights_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._weights_loaded

    def predict_frame(self, image: np.ndarray) -> FramePredictionResponse:
        """
        Predict helmet detections and contacts for a single frame.

        Args:
            image: Input image in BGR format (H, W, C)

        Returns:
            FramePredictionResponse with detections and contact predictions

        Raises:
            RuntimeError: If model not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Run YOLO inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        # Parse detections
        helmets = self._parse_detections(results[0])

        # Predict contacts between helmet pairs
        contacts = self._predict_contacts(helmets)

        # Determine if frame has contact
        frame_has_contact = any(c.contact_probability > 0.5 for c in contacts)

        inference_time_ms = (time.time() - start_time) * 1000

        return FramePredictionResponse(
            helmets=helmets,
            contacts=contacts,
            frame_has_contact=frame_has_contact,
            inference_time_ms=inference_time_ms
        )

    def predict_clip(
        self,
        frames: List[np.ndarray],
        max_frames: int = 30
    ) -> ClipPredictionResponse:
        """
        Predict helmet contacts for a video clip.

        Args:
            frames: List of frames in BGR format
            max_frames: Maximum frames to analyze (for efficiency)

        Returns:
            ClipPredictionResponse with aggregated predictions

        Raises:
            RuntimeError: If model not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Sample frames if needed
        sampled_frames, frame_indices = sample_frames(frames, max_frames)

        # Run inference on each frame
        contact_frames = []
        max_contact_prob = 0.0
        total_helmets = 0

        for idx, frame in zip(frame_indices, sampled_frames):
            result = self.predict_frame(frame)

            if result.frame_has_contact:
                contact_frames.append(idx)

            # Track max contact probability
            for contact in result.contacts:
                max_contact_prob = max(max_contact_prob, contact.contact_probability)

            total_helmets += len(result.helmets)

        avg_helmets = total_helmets / len(sampled_frames) if sampled_frames else 0.0
        inference_time_ms = (time.time() - start_time) * 1000

        return ClipPredictionResponse(
            total_frames=len(frames),
            frames_analyzed=len(sampled_frames),
            contact_frames=contact_frames,
            max_contact_probability=max_contact_prob,
            average_helmets_per_frame=avg_helmets,
            inference_time_ms=inference_time_ms
        )

    def _parse_detections(self, result) -> List[HelmetDetection]:
        """
        Parse YOLO results into HelmetDetection objects.

        Args:
            result: YOLO result object

        Returns:
            List of HelmetDetection objects
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) xyxy format
        confidences = result.boxes.conf.cpu().numpy()  # (N,)
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None

        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box

            detection = HelmetDetection(
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                confidence=float(conf),
                class_name="helmet",
                helmet_id=idx
            )
            detections.append(detection)

        return detections

    def _predict_contacts(self, helmets: List[HelmetDetection]) -> List[ContactPrediction]:
        """
        Predict contacts between helmet pairs using geometric heuristics.

        Contact probability is based on:
        - IoU (intersection over union)
        - Distance between helmet centers
        - Overlap area

        Args:
            helmets: List of detected helmets

        Returns:
            List of ContactPrediction objects
        """
        contacts = []

        # Check all pairs of helmets
        for i in range(len(helmets)):
            for j in range(i + 1, len(helmets)):
                h1 = helmets[i]
                h2 = helmets[j]

                # Get bounding boxes
                box1 = (h1.bbox.x1, h1.bbox.y1, h1.bbox.x2, h1.bbox.y2)
                box2 = (h2.bbox.x1, h2.bbox.y1, h2.bbox.x2, h2.bbox.y2)

                # Calculate IoU
                iou = calculate_iou(box1, box2)

                # Calculate center distance
                center1 = calculate_box_center(box1)
                center2 = calculate_box_center(box2)
                distance = calculate_euclidean_distance(center1, center2)

                # Calculate contact probability using heuristics
                contact_prob = self._calculate_contact_probability(iou, distance)

                # Only include if probability is above threshold
                if contact_prob > 0.1:
                    contacts.append(
                        ContactPrediction(
                            helmet_1_id=h1.helmet_id,
                            helmet_2_id=h2.helmet_id,
                            contact_probability=contact_prob,
                            distance=float(distance),
                            overlap_iou=float(iou)
                        )
                    )

        return contacts

    def _calculate_contact_probability(self, iou: float, distance: float) -> float:
        """
        Calculate contact probability from geometric features.

        This is a heuristic approach. In production, this would be replaced
        with a trained classifier using temporal features, velocity, etc.

        Args:
            iou: Intersection over Union
            distance: Distance between centers

        Returns:
            Contact probability [0, 1]
        """
        # IoU component (high IoU = likely contact)
        iou_score = min(iou / self.iou_threshold, 1.0)

        # Distance component (low distance = likely contact)
        distance_score = max(0, 1.0 - distance / self.distance_threshold)

        # Weighted combination
        contact_prob = 0.6 * iou_score + 0.4 * distance_score

        return min(max(contact_prob, 0.0), 1.0)
