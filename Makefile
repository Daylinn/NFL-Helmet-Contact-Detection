.PHONY: help install download-data build-cache train evaluate export-onnx api streamlit test lint format clean docker-build docker-run

# Default target
help:
	@echo "NFL Helmet Impact Detection - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install dependencies"
	@echo "  make download-data   Download Kaggle dataset"
	@echo ""
	@echo "Data & Training:"
	@echo "  make build-cache     Build preprocessed crops cache"
	@echo "  make build-cache-tiny  Build cache in tiny mode (fast)"
	@echo "  make train           Train model"
	@echo "  make train-tiny      Train in tiny mode (fast)"
	@echo "  make evaluate        Evaluate trained model"
	@echo "  make export-onnx     Export model to ONNX"
	@echo ""
	@echo "Inference:"
	@echo "  make api             Run FastAPI server"
	@echo "  make streamlit       Run Streamlit UI"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run tests"
	@echo "  make lint            Lint code with ruff"
	@echo "  make format          Format code with ruff"
	@echo "  make clean           Clean generated files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container"

# Installation
install:
	pip install --upgrade pip
	pip install -e .[dev]

# Data pipeline
download-data:
	python scripts/download_data.py

build-cache:
	python scripts/build_crops_cache.py --config configs/base.yaml

build-cache-tiny:
	python scripts/build_crops_cache.py --config configs/base.yaml --override-config configs/tiny.yaml

# Training
train:
	python scripts/train.py --config configs/base.yaml

train-tiny:
	python scripts/train.py --config configs/base.yaml --override-config configs/tiny.yaml

# Evaluation
evaluate:
	python scripts/evaluate.py --config configs/base.yaml

export-onnx:
	python scripts/export_onnx.py --config configs/base.yaml --verify

# Inference
api:
	python -m src.impact_detector.api

streamlit:
	streamlit run app.py

# Development
test:
	pytest tests/ -v --cov=src/impact_detector

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/ .ruff_cache/ htmlcov/ dist/ build/ *.egg-info

# Docker
docker-build:
	docker build -t nfl-impact-detector:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models nfl-impact-detector:latest
