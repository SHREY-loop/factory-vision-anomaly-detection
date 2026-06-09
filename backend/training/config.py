"""
Shared configuration for anomaly detection training and inference.

All tunable hyperparameters live here so they can be adjusted without
touching business logic in any other module.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DIR: Path = BACKEND_ROOT / "models"
DEFAULT_PREDICTIONS_DIR: Path = BACKEND_ROOT / "predictions"
DEFAULT_REPORTS_DIR: Path = BACKEND_ROOT / "reports"

# Part type identifiers must match: part_A, part_B, part_C, etc.
PART_TYPE_PATTERN = re.compile(r"^part_[A-Za-z0-9]+$")

# ---------------------------------------------------------------------------
# Per-part path helpers
# ---------------------------------------------------------------------------


def validate_part_type(part_type: str) -> str:
    """Return *part_type* if valid, else raise ValueError."""
    normalized = part_type.strip()
    if not PART_TYPE_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Invalid part_type '{part_type}'. "
            "Expected format: part_A, part_B, part_C (letters, digits, underscores)."
        )
    return normalized


def get_part_dataset_dir(part_type: str) -> Path:
    """Return the OK image folder for a given part type.

    Example: get_part_dataset_dir("part_A") -> datasets/part_A/ok/
    """
    return BACKEND_ROOT / "datasets" / validate_part_type(part_type) / "ok"


def get_part_model_dir(part_type: str) -> Path:
    """Return the model output directory for a given part type.

    Example: get_part_model_dir("part_B") -> models/part_B/
    """
    return DEFAULT_MODEL_DIR / validate_part_type(part_type)


def get_part_model_path(part_type: str) -> Path:
    """Return the .pkl model file path for a given part type.

    Example: get_part_model_path("part_C") -> models/part_C/anomaly_model.pkl
    """
    return get_part_model_dir(part_type) / "anomaly_model.pkl"


def get_part_report_path(part_type: str) -> Path:
    """Return the training report path for a given part type.

    Example: get_part_report_path("part_A") -> models/part_A/training_report.json
    """
    return get_part_model_dir(part_type) / "training_report.json"


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

INPUT_SIZE: int = 224  # EfficientNet-B0 native input resolution

# ImageNet statistics — matches torchvision EfficientNet pretrained weights
NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
NORMALIZE_STD: list[float] = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_DIM: int = 1280  # output channels of EfficientNet-B0 features block

# ---------------------------------------------------------------------------
# Memory bank / anomaly scoring
# ---------------------------------------------------------------------------

# Fraction of extracted embeddings to keep in the memory bank.
# 1.0 = keep all (recommended for small datasets).
CORESET_RATIO: float = 1.0

# Percentile of per-image anomaly scores used to set the decision threshold.
THRESHOLD_PERCENTILE: float = 95.0

# ---------------------------------------------------------------------------
# Training defaults (CLI overrides accepted)
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEED: int = 42
DEFAULT_NUM_WORKERS: int = 0  # 0 = safe on Windows

# ---------------------------------------------------------------------------
# Training augmentation (in-memory only — source images are never written)
# ---------------------------------------------------------------------------

AUGMENT_ENABLED: bool = True
AUGMENT_COPIES_PER_IMAGE: int = 1

# ---------------------------------------------------------------------------
# API limits
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
