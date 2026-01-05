"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.impact_detector.api import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_no_file():
    """Test predict endpoint without file."""
    response = client.post("/predict")

    assert response.status_code == 422  # Validation error


def test_predict_invalid_file_type():
    """Test predict endpoint with invalid file type."""
    files = {"file": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


# Note: Full prediction tests would require a trained model and test video
# which are not available in CI environment
