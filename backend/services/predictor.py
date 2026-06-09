"""
Anomaly detection inference service.

Stateless design — accepts raw image bytes from any source:
  - HTTP upload (current)
  - Camera frame (future)
  - Video frame (future)

No source-specific logic lives here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from model.efficientnet import FeatureExtractor, load_anomaly_model
from training.config import DEFAULT_PREDICTIONS_DIR, get_part_model_path

logger = logging.getLogger(__name__)

PredictionResult = dict[str, Any]

_extractor: FeatureExtractor | None = None


def _get_extractor() -> FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def predict(image_bytes: bytes, part_type: str) -> PredictionResult:
    """
    Analyse a manufactured part image and return anomaly detection result.

    Parameters
    ----------
    image_bytes:
        Raw bytes of a JPEG/PNG image from any source (upload, camera, video).
    part_type:
        Part type identifier (e.g. "part_A", "part_B", "part_C").

    Returns
    -------
    dict with keys:
        status, confidence, anomaly_score, raw_distance, threshold,
        part_type, defect_type, bounding_boxes, heatmap
    """
    model_path = get_part_model_path(part_type)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model for part type '{part_type}'.\n"
            f"Expected file: {model_path}\n"
            f"Train it first:\n"
            f"  cd backend && python -m training.train --part-type {part_type}"
        )

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Unable to decode image bytes: {exc}\n"
            "Supported formats: JPEG, PNG."
        ) from exc

    extractor = _get_extractor()
    embedding = extractor.extract(pil_image)

    detector = load_anomaly_model(part_type=part_type)
    inference = detector.predict(embedding)

    logger.debug(
        "Inference part_type=%s model=%s threshold=%.6f raw_distance=%.6f status=%s",
        part_type,
        model_path,
        inference["threshold"],
        inference["raw_distance"],
        inference["status"],
    )

    result: PredictionResult = {
        "status": inference["status"],
        "confidence": inference["confidence"],
        "anomaly_score": inference["anomaly_score"],
        "raw_distance": inference["raw_distance"],
        "threshold": inference["threshold"],
        "part_type": part_type,
        "defect_type": None,
        "bounding_boxes": [],
        "heatmap": None,
    }

    try:
        _save_prediction_history(pil_image, result, part_type=part_type)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to save prediction history: %s", exc)

    logger.info(
        "Prediction [%s] — status: %s  confidence: %.1f%%  anomaly_score: %.4f",
        part_type,
        result["status"],
        result["confidence"],
        result["anomaly_score"],
    )

    return result


def get_loaded_parts() -> list[str]:
    """Return part types currently loaded in the in-process model cache."""
    from model.efficientnet import _detector_cache

    return list(_detector_cache.keys())


def _save_prediction_history(
    pil_image: Image.Image,
    result: PredictionResult,
    *,
    part_type: str,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
) -> None:
    """Save inference image and metadata under predictions/<part_type>/."""
    save_dir = predictions_dir / part_type
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    base_path = save_dir / timestamp

    image_path = base_path.with_suffix(".jpg")
    pil_image.save(image_path, format="JPEG", quality=90)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "part_type": part_type,
        "status": result["status"],
        "confidence": result["confidence"],
        "anomaly_score": result["anomaly_score"],
        "raw_distance": result["raw_distance"],
        "threshold": result["threshold"],
        "image_file": image_path.name,
    }
    json_path = base_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.debug("Saved prediction history: %s", base_path)
