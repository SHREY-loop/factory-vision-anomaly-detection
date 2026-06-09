"""
EfficientNet-B0 feature extractor and anomaly detection core.

Architecture:
    FeatureExtractor  — wraps EfficientNet-B0 backbone, produces 1280-dim embeddings
    AnomalyDetector   — memory bank of OK embeddings + calibrated threshold
    save/load helpers — pickle-based persistence with a singleton in-process cache
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from training.config import (
    DEFAULT_MODEL_DIR,
    FEATURE_DIM,
    INPUT_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    get_part_model_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process singleton cache — avoids reloading on every request (keyed by part_type).
# ---------------------------------------------------------------------------

_detector_cache: dict[str, AnomalyDetector] = {}
_detector_cache_mtime: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------


class FeatureExtractor(nn.Module):
    """
    EfficientNet-B0 backbone used purely as a feature extractor.

    The classifier head is discarded.  The ``features`` block (convolutional
    backbone) is kept frozen with ImageNet pretrained weights.  Global
    average pooling reduces spatial dims → (batch, 1280) embeddings.

    Usage
    -----
    extractor = FeatureExtractor(device)
    embedding = extractor.extract(pil_image)   # → np.ndarray (1280,)
    embeddings = extractor.extract_batch(dataloader)  # → np.ndarray (N, 1280)
    """

    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Keep only the convolutional feature block; discard classifier
        self._features: nn.Sequential = backbone.features
        self._pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Freeze weights — we use pretrained ImageNet features as-is
        for param in self.parameters():
            param.requires_grad_(False)

        self.to(self.device)
        self.eval()

        logger.info(
            "FeatureExtractor ready on %s (output dim: %d)", self.device, FEATURE_DIM
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (batch, 1280) embedding tensor."""
        x = x.to(self.device)
        with torch.no_grad():
            feat_map = self._features(x)           # (B, 1280, H', W')
            pooled = self._pool(feat_map)           # (B, 1280, 1, 1)
            embedding = pooled.flatten(start_dim=1) # (B, 1280)
        return embedding

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def extract(self, pil_image: Image.Image) -> NDArray[np.float32]:
        """Extract a single embedding from a PIL image."""
        transform = _get_inference_transform()
        tensor = transform(pil_image).unsqueeze(0)   # (1, 3, H, W)
        embedding = self(tensor)
        return embedding.cpu().numpy().squeeze(0)    # (1280,)

    @torch.no_grad()
    def extract_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> NDArray[np.float32]:
        """
        Extract embeddings from an entire DataLoader.

        Parameters
        ----------
        dataloader:
            Yields (image_tensor_batch, paths) tuples (from OKImageDataset).

        Returns
        -------
        embeddings: np.ndarray of shape (N, 1280)
        """
        all_embeddings: list[NDArray[np.float32]] = []

        for batch_tensors, _ in dataloader:
            batch_embeddings = self(batch_tensors)              # (B, 1280)
            all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)    # (N, 1280)


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """
    Memory bank anomaly detector.

    Stores a bank of embeddings extracted from OK training images.
    At inference, computes the minimum L2 distance from a query embedding
    to any bank vector.  The score is normalised to [0, 1] using the
    calibrated training distance range.

    Parameters
    ----------
    memory_bank:
        np.ndarray of shape (N, D) — OK training embeddings.
    threshold:
        Raw distance threshold separating OK from NOT_OK.
    max_distance:
        Maximum training distance used for score normalisation.
    """

    def __init__(
        self,
        memory_bank: NDArray[np.float32],
        threshold: float,
        max_distance: float,
    ) -> None:
        self.memory_bank: NDArray[np.float32] = memory_bank.astype(np.float32)
        self.threshold: float = float(threshold)
        self.max_distance: float = float(max_distance)

        logger.info(
            "AnomalyDetector initialised — bank size: %d, threshold: %.4f",
            len(self.memory_bank),
            self.threshold,
        )

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_embedding(self, embedding: NDArray[np.float32]) -> float:
        """
        Compute nearest-neighbour L2 distance from *embedding* to memory bank.

        Returns raw distance (not normalised).
        """
        diff = self.memory_bank - embedding[np.newaxis, :]  # (N, D)
        distances = np.linalg.norm(diff, axis=1)            # (N,)
        return float(distances.min())

    def normalise_score(self, raw_distance: float) -> float:
        """
        Map raw L2 distance -> [0, 1].

        Reference scale: 2 x threshold

        - distance = 0            -> score = 0.00  (identical to a training image)
        - distance = threshold    -> score = 0.50  (exactly at decision boundary)
        - distance >= 2*threshold -> score = 1.00  (clearly anomalous)

        Using 2*threshold (not max_distance) prevents the score from clipping
        to 1.0 for moderately anomalous images and keeps the full [0,1] range
        meaningful across the decision boundary.
        """
        ref = max(2.0 * self.threshold, 1e-9)
        return float(np.clip(raw_distance / ref, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, embedding: NDArray[np.float32]) -> dict[str, Any]:
        """
        Classify a single embedding.

        Returns
        -------
        dict with keys:
            status         : "OK" | "NOT_OK"
            confidence     : float  (0–100 %)
            anomaly_score  : float  (0.0–1.0, normalised)
            raw_distance   : float  (unnormalised, for diagnostics)
        """
        raw_distance = self.score_embedding(embedding)
        anomaly_score = self.normalise_score(raw_distance)
        is_anomaly = raw_distance > self.threshold

        status = "NOT_OK" if is_anomaly else "OK"

        # Confidence: fraction of one threshold-width away from the boundary,
        # capped at 1.0 so confidence maxes at 100%.
        #
        #   At boundary (raw == threshold): deviation=0 -> confidence=50%
        #   At distance=0 (perfect match):  deviation=1 -> confidence=100%
        #   At distance=2*threshold:        deviation=1 -> confidence=100%
        deviation = abs(raw_distance - self.threshold) / max(self.threshold, 1e-9)
        deviation = min(deviation, 1.0)          # cap so confidence <= 100%
        confidence = 50.0 + 50.0 * deviation

        return {
            "status": status,
            "confidence": round(confidence, 2),
            "anomaly_score": round(anomaly_score, 4),
            "raw_distance": round(raw_distance, 6),
            "threshold": round(self.threshold, 6),
        }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_anomaly_model(
    detector: AnomalyDetector,
    path: Path | str,
) -> None:
    """Serialise AnomalyDetector to disk using pickle."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        pickle.dump(detector, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved anomaly model to %s", dest)


def load_anomaly_model(
    part_type: str,
    *,
    force_reload: bool = False,
) -> AnomalyDetector:
    """
    Load AnomalyDetector from ``models/<part_type>/anomaly_model.pkl``.

    Results are cached in-process per part_type. Pass ``force_reload=True``
    to bypass the cache.
    """
    global _detector_cache, _detector_cache_mtime

    model_path = get_part_model_path(part_type)
    file_mtime = model_path.stat().st_mtime if model_path.is_file() else 0.0

    if not force_reload and part_type in _detector_cache:
        cached_mtime = _detector_cache_mtime.get(part_type)
        if cached_mtime == file_mtime:
            return _detector_cache[part_type]
        logger.info(
            "Model file changed on disk — reloading (part_type: %s, path: %s)",
            part_type,
            model_path,
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found for part type '{part_type}': {model_path}\n"
            f"Train it first:\n"
            f"  cd backend && python -m training.train --part-type {part_type}"
        )

    with open(model_path, "rb") as fh:
        detector: AnomalyDetector = pickle.load(fh)

    _detector_cache[part_type] = detector
    _detector_cache_mtime[part_type] = file_mtime
    logger.info(
        "Loaded anomaly model from %s (part_type: %s, bank_size: %d, threshold: %.6f)",
        model_path.resolve(),
        part_type,
        len(detector.memory_bank),
        detector.threshold,
    )
    return detector


def get_available_part_types() -> list[str]:
    """
    Scan the models/ directory and return part types that have a trained model.

    A part type is considered available when
    ``models/<part_type>/anomaly_model.pkl`` exists.
    Returns a sorted list of part_type strings.
    """
    if not DEFAULT_MODEL_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in DEFAULT_MODEL_DIR.iterdir()
        if d.is_dir() and (d / "anomaly_model.pkl").is_file()
    )


def clear_model_cache(part_type: str | None = None) -> None:
    """Release the in-process model cache.

    Parameters
    ----------
    part_type:
        If given, clears only that part's cached detector.
        If None, clears all cached detectors.
    """
    global _detector_cache, _detector_cache_mtime
    if part_type is None:
        _detector_cache = {}
        _detector_cache_mtime = {}
        logger.debug("All anomaly model caches cleared")
    else:
        cache_key = part_type
        if cache_key in _detector_cache:
            del _detector_cache[cache_key]
            _detector_cache_mtime.pop(cache_key, None)
            logger.debug("Anomaly model cache cleared for part_type=%s", part_type)


# ---------------------------------------------------------------------------
# Inference preprocessing
# ---------------------------------------------------------------------------


def _get_inference_transform() -> transforms.Compose:
    """Deterministic transform matching training preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
