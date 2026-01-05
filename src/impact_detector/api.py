"""FastAPI service for helmet impact detection."""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.impact_detector.config import load_config
from src.impact_detector.inference import run_inference

# Load config
config = load_config()

# Create temp directory
temp_dir = Path(config.api.temp_dir)
temp_dir.mkdir(parents=True, exist_ok=True)

# Create app
app = FastAPI(
    title="NFL Helmet Impact Detection API",
    description="API for detecting helmet impacts in NFL game footage",
    version="0.1.0",
)


class Detection(BaseModel):
    """Impact detection result."""

    frame: int
    time_sec: float
    bbox: List[int]
    score: float


class PredictionResponse(BaseModel):
    """Prediction response."""

    detections: List[Detection]
    num_detections: int
    video_filename: Optional[str] = None
    annotated_video_path: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NFL Helmet Impact Detection API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check if model exists
    model_path = config.inference.onnx_path if config.inference.use_onnx else config.inference.model_path

    model_exists = Path(model_path).exists()

    return {
        "status": "healthy" if model_exists else "degraded",
        "model_loaded": model_exists,
        "model_path": model_path,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = None,
    min_score: Optional[float] = None,
    annotate: bool = False,
):
    """Predict helmet impacts in uploaded video.

    Args:
        file: Uploaded video file
        sample_rate: Process every N frames (default from config)
        min_score: Minimum impact confidence (default from config)
        annotate: Whether to return annotated video

    Returns:
        Detection results and optional annotated video
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.api.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config.api.allowed_extensions}",
        )

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=temp_dir) as tmp:
        content = await file.read()

        # Check file size
        if len(content) > config.api.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {config.api.max_upload_size} bytes",
            )

        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Override config if needed
        if sample_rate is not None:
            config.inference.sample_rate = sample_rate
        if min_score is not None:
            config.inference.min_score = min_score

        # Prepare output path for annotated video
        output_path = None
        if annotate:
            output_path = str(temp_dir / f"annotated_{Path(tmp_path).name}")

        # Run inference
        detections, annotated_path = run_inference(
            tmp_path,
            config,
            annotate=annotate,
            output_path=output_path,
        )

        # Format response
        response = {
            "detections": detections,
            "num_detections": len(detections),
            "video_filename": file.filename,
        }

        if annotate and annotated_path:
            response["annotated_video_path"] = annotated_path

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Clean up uploaded file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/download/{filename}")
async def download_annotated(filename: str):
    """Download annotated video.

    Args:
        filename: Filename of annotated video

    Returns:
        Video file
    """
    file_path = temp_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename,
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    print("Starting NFL Helmet Impact Detection API...")
    print(f"Config: {config}")
    print(f"Temp directory: {temp_dir}")

    # Verify model exists
    model_path = config.inference.onnx_path if config.inference.use_onnx else config.inference.model_path

    if not Path(model_path).exists():
        print(f"WARNING: Model not found at {model_path}")
        print("Please train a model or update the config")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    print("Shutting down API...")

    # Optional: Clean up temp files
    # for file in temp_dir.glob("*"):
    #     file.unlink()


def main():
    """Run API server."""
    import uvicorn

    port = int(os.environ.get("PORT", config.api.port))
    host = os.environ.get("HOST", config.api.host)

    uvicorn.run(
        "src.impact_detector.api:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
