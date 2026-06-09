"""
Unsupervised anomaly detection training pipeline.

Algorithm
---------
1. Load all OK images from dataset/ok/
2. Extract 1280-dim embeddings via frozen EfficientNet-B0 backbone
3. Optionally subsample embeddings into a coreset (KMeans) if CORESET_RATIO < 1.0
4. Compute pairwise nearest-neighbour distances on the full training set
5. Calibrate decision threshold at THRESHOLD_PERCENTILE of training distances
6. Save AnomalyDetector (memory bank + threshold) → models/<part_type>/anomaly_model.pkl
7. Write training diagnostics → models/<part_type>/training_report.json

Usage (from backend/ directory)
---------------------------------
    python -m training.train --part-type part_A
    python -m training.train --part-type part_B --threshold-percentile 90
    python -m training.train --part-type part_C --coreset-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from model.efficientnet import (
    AnomalyDetector,
    FeatureExtractor,
    save_anomaly_model,
)
from training.augmentation import augmentation_spec
from training.config import (
    AUGMENT_COPIES_PER_IMAGE,
    AUGMENT_ENABLED,
    CORESET_RATIO,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    FEATURE_DIM,
    THRESHOLD_PERCENTILE,
    get_part_dataset_dir,
    get_part_model_path,
    get_part_report_path,
    validate_part_type,
)
from training.dataset import create_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train anomaly detection model on OK images only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--part-type",
        type=str,
        required=True,
        help=(
            "Part type identifier (e.g. part_A, part_B, part_C). "
            "Resolves dataset to datasets/<part_type>/ok/ and "
            "model output to models/<part_type>/."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to flat folder containing OK images (overrides --part-type default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for trained anomaly model (.pkl) (overrides --part-type default)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Destination path for training report (.json) (overrides --part-type default)",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=THRESHOLD_PERCENTILE,
        help="Percentile of training distances used to set decision threshold (0–100)",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=CORESET_RATIO,
        help="Fraction of embeddings to keep in memory bank (1.0 = keep all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes (use 0 on Windows)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=AUGMENT_COPIES_PER_IMAGE,
        help=(
            "Number of in-memory augmented copies per source image. "
            "Total embeddings per image = 1 original + this value."
        ),
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentation (original images only)",
    )

    args = parser.parse_args()
    args.part_type = validate_part_type(args.part_type)

    if args.data_dir is None:
        args.data_dir = get_part_dataset_dir(args.part_type)
    if args.output is None:
        args.output = get_part_model_path(args.part_type)
    if args.report is None:
        args.report = get_part_report_path(args.part_type)

    return args


# ---------------------------------------------------------------------------
# Coreset subsampling
# ---------------------------------------------------------------------------


def _subsample_coreset(
    embeddings: np.ndarray,
    ratio: float,
    seed: int,
) -> np.ndarray:
    """
    Reduce the memory bank using KMeans cluster centres.

    Parameters
    ----------
    embeddings : (N, D) float32 array
    ratio      : fraction to keep, e.g. 0.1 keeps ~10 % of N
    seed       : random state for reproducibility

    Returns
    -------
    coreset : (K, D) float32 array where K = max(1, int(N * ratio))
    """
    from sklearn.cluster import MiniBatchKMeans  # lazy import

    n_clusters = max(1, int(len(embeddings) * ratio))
    logger.info(
        "Coreset subsampling: %d → %d vectors (ratio=%.2f)",
        len(embeddings),
        n_clusters,
        ratio,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=3,
        batch_size=min(1024, len(embeddings)),
    )
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_.astype(np.float32)


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------


def _compute_training_distances(memory_bank: np.ndarray) -> np.ndarray:
    """
    For each embedding in the memory bank compute its nearest-neighbour
    distance within the bank itself.

    Returns an (N,) array of minimum distances (excluding self).
    """
    logger.info(
        "Computing nearest-neighbour distances for %d embeddings …", len(memory_bank)
    )

    # Batch computation to avoid OOM on large banks
    batch = 256
    distances = np.full(len(memory_bank), np.inf, dtype=np.float32)

    for start in range(0, len(memory_bank), batch):
        end = min(start + batch, len(memory_bank))
        query = memory_bank[start:end]                      # (B, D)
        diff = memory_bank[np.newaxis, :, :] - query[:, np.newaxis, :]  # (B, N, D)
        dists = np.linalg.norm(diff, axis=2)               # (B, N)
        # Exclude self-distances (diagonal of the submatrix)
        for i in range(end - start):
            dists[i, start + i] = np.inf
        distances[start:end] = dists.min(axis=1)

    return distances


# ---------------------------------------------------------------------------
# Training report
# ---------------------------------------------------------------------------


def _write_training_report(
    report_path: Path,
    total_images: int,
    feature_vectors_extracted: int,
    memory_bank_size: int,
    threshold: float,
    max_distance: float,
    feature_dimension: int,
    elapsed_seconds: float,
    coreset_ratio: float,
    threshold_percentile: float,
    *,
    augmentation_enabled: bool,
    augment_copies_per_image: int,
    embeddings_per_image: int,
    original_embeddings: int,
    augmented_embeddings: int,
    memory_bank_size_before_coreset: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_images": total_images,
        "feature_vectors_extracted": feature_vectors_extracted,
        "memory_bank_size": memory_bank_size,
        "threshold": round(threshold, 6),
        "max_distance": round(max_distance, 6),
        "feature_dimension": feature_dimension,
        "coreset_ratio": coreset_ratio,
        "threshold_percentile": threshold_percentile,
        "training_time_seconds": round(elapsed_seconds, 2),
        "augmentation": {
            "enabled": augmentation_enabled,
            "copies_per_image": augment_copies_per_image,
            "embeddings_per_image": embeddings_per_image,
            "original_embeddings": original_embeddings,
            "augmented_embeddings": augmented_embeddings,
            "memory_bank_size_before_coreset": memory_bank_size_before_coreset,
            "parameters": augmentation_spec() if augmentation_enabled else {},
        },
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Training report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("Anomaly Detection Training — Phase 1")
    logger.info("=" * 60)
    logger.info("Part type:            %s", args.part_type)
    logger.info("Dataset:              %s", args.data_dir)
    logger.info("Output model:         %s", args.output)
    logger.info("Training report:      %s", args.report)
    logger.info("Threshold percentile: %.1f%%", args.threshold_percentile)
    logger.info("Coreset ratio:        %.2f", args.coreset_ratio)

    augment_enabled = not args.no_augment and args.augment_copies > 0
    embeddings_per_image = 1 + args.augment_copies if augment_enabled else 1
    logger.info(
        "Augmentation:         %s",
        f"enabled ({args.augment_copies} copies/image, "
        f"{embeddings_per_image} embeddings/image)"
        if augment_enabled
        else "disabled",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device:               %s", device)
    logger.info("=" * 60)

    start_time = time.time()

    # ------------------------------------------------------------------ #
    # Step 1 — Load dataset                                               #
    # ------------------------------------------------------------------ #
    loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=augment_enabled,
        augment_copies=args.augment_copies,
    )
    total_images = len(loader.dataset.image_paths)  # type: ignore[attr-defined]
    training_samples = len(loader.dataset)
    logger.info("Loaded %d OK images for feature extraction", total_images)
    if augment_enabled:
        logger.info(
            "Training samples (original + augmented): %d",
            training_samples,
        )

    # ------------------------------------------------------------------ #
    # Step 2 — Extract embeddings                                         #
    # ------------------------------------------------------------------ #
    extractor = FeatureExtractor(device=device)
    logger.info("Extracting features …")
    embeddings = extractor.extract_batch(loader)  # (N, 1280)
    feature_vectors_extracted = len(embeddings)
    original_embeddings = total_images
    augmented_embeddings = feature_vectors_extracted - total_images
    memory_bank_size_before_coreset = feature_vectors_extracted
    logger.info(
        "Extracted %d embeddings of dimension %d "
        "(%d original + %d augmented)",
        feature_vectors_extracted,
        embeddings.shape[1],
        original_embeddings,
        augmented_embeddings,
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Optional coreset subsampling                               #
    # ------------------------------------------------------------------ #
    if args.coreset_ratio < 1.0:
        memory_bank = _subsample_coreset(embeddings, args.coreset_ratio, args.seed)
    else:
        logger.info(
            "Coreset ratio=1.0 — keeping all %d embeddings", feature_vectors_extracted
        )
        memory_bank = embeddings

    memory_bank_size = len(memory_bank)
    if memory_bank_size_before_coreset != memory_bank_size:
        logger.info(
            "Memory bank after coreset: %d (was %d)",
            memory_bank_size,
            memory_bank_size_before_coreset,
        )

    # ------------------------------------------------------------------ #
    # Step 4 — Calibrate threshold                                        #
    # ------------------------------------------------------------------ #
    logger.info("Calibrating anomaly threshold …")
    training_distances = _compute_training_distances(memory_bank)
    threshold = float(np.percentile(training_distances, args.threshold_percentile))
    max_distance = float(training_distances.max())

    logger.info(
        "Distance stats — min: %.4f  mean: %.4f  max: %.4f  p%.0f: %.4f",
        training_distances.min(),
        training_distances.mean(),
        max_distance,
        args.threshold_percentile,
        threshold,
    )
    logger.info("Decision threshold: %.4f", threshold)

    # ------------------------------------------------------------------ #
    # Step 5 — Save model                                                 #
    # ------------------------------------------------------------------ #
    detector = AnomalyDetector(
        memory_bank=memory_bank,
        threshold=threshold,
        max_distance=max_distance,
    )
    save_anomaly_model(detector, path=args.output)
    logger.info("Anomaly model saved to %s", args.output)

    # ------------------------------------------------------------------ #
    # Step 6 — Write training report                                      #
    # ------------------------------------------------------------------ #
    elapsed = time.time() - start_time
    _write_training_report(
        report_path=args.report,
        total_images=total_images,
        feature_vectors_extracted=feature_vectors_extracted,
        memory_bank_size=memory_bank_size,
        threshold=threshold,
        max_distance=max_distance,
        feature_dimension=FEATURE_DIM,
        elapsed_seconds=elapsed,
        coreset_ratio=args.coreset_ratio,
        threshold_percentile=args.threshold_percentile,
        augmentation_enabled=augment_enabled,
        augment_copies_per_image=args.augment_copies if augment_enabled else 0,
        embeddings_per_image=embeddings_per_image,
        original_embeddings=original_embeddings,
        augmented_embeddings=augmented_embeddings,
        memory_bank_size_before_coreset=memory_bank_size_before_coreset,
    )

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Training complete in %.1fs", elapsed)
    logger.info("  Images processed  : %d", total_images)
    logger.info("  Embeddings/image  : %d", embeddings_per_image)
    logger.info(
        "  Embeddings stored : %d × %d (before coreset: %d)",
        memory_bank_size,
        FEATURE_DIM,
        memory_bank_size_before_coreset,
    )
    logger.info("  Threshold         : %.4f", threshold)
    logger.info("  Model             : %s", args.output)
    logger.info("  Report            : %s", args.report)
    logger.info("")
    logger.info("Next step — run inference:")
    logger.info("  cd backend && uvicorn main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
