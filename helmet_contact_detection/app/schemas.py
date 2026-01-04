"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates in xyxy format."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class HelmetDetection(BaseModel):
    """Single helmet detection result."""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_name: str = Field(default="helmet", description="Object class")
    helmet_id: Optional[int] = Field(None, description="Tracking ID if available")


class ContactPrediction(BaseModel):
    """Contact/impact prediction for a pair of helmets."""
    helmet_1_id: int
    helmet_2_id: int
    contact_probability: float = Field(..., ge=0.0, le=1.0)
    distance: float = Field(..., description="Distance between helmet centers (pixels)")
    overlap_iou: float = Field(..., ge=0.0, le=1.0, description="Intersection over Union")


class FramePredictionResponse(BaseModel):
    """Response for single frame prediction."""
    helmets: List[HelmetDetection] = Field(default_factory=list)
    contacts: List[ContactPrediction] = Field(default_factory=list)
    frame_has_contact: bool = Field(..., description="Whether any contact detected in frame")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class ClipPredictionResponse(BaseModel):
    """Response for video clip prediction."""
    total_frames: int
    frames_analyzed: int
    contact_frames: List[int] = Field(default_factory=list, description="Frame indices with contact")
    max_contact_probability: float = Field(..., ge=0.0, le=1.0)
    average_helmets_per_frame: float
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    weights_loaded: bool
    message: str
    version: str = Field(default="1.0.0")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
