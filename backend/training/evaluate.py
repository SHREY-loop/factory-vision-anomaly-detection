"""
Evaluate a trained anomaly detection model.

Reports anomaly score statistics for the training OK images.
Optionally compares against a separate NOT_OK evaluation folder.

Usage (from backend/ directory)
---------------------------------
    python -m training.evaluate --part-type part_A
    python -m training.evaluate --part-type part_B --not-ok-dir datasets/part_B/not_ok
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from model.efficientnet import FeatureExtractor, load_anomaly_model
from training.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    get_part_dataset_dir,
    get_part_model_path,
    validate_part_type,
)
from training.dataset import OKImageDataset, build_eval_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection model on OK (and optionally NOT_OK) images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--part-type",
        type=str,
        required=True,
        help="Part type identifier (e.g. part_A, part_B, part_C)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Flat folder of OK images (default: datasets/<part_type>/ok/)",
    )
    parser.add_argument(
        "--not-ok-dir",
        type=Path,
        default=None,
        help="Optional flat folder of NOT_OK images for separation statistics",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to anomaly_model.pkl (default: models/<part_type>/anomaly_model.pkl)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    args.part_type = validate_part_type(args.part_type)
    if args.data_dir is None:
        args.data_dir = get_part_dataset_dir(args.part_type)
    if args.model_path is None:
        args.model_path = get_part_model_path(args.part_type)

    return args


def _score_folder(
    extractor: FeatureExtractor,
    folder: Path,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Extract embeddings from *folder*."""
    from torch.utils.data import DataLoader

    dataset = OKImageDataset(data_dir=folder, transform=build_eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return extractor.extract_batch(loader)


def _print_score_stats(scores: np.ndarray, label: str, threshold: float) -> None:
    below = int((scores <= threshold).sum())
    above = int((scores > threshold).sum())
    logger.info(
        "  %-12s — n=%d  min=%.4f  mean=%.4f  max=%.4f  std=%.4f",
        label,
        len(scores),
        scores.min(),
        scores.mean(),
        scores.max(),
        scores.std(),
    )
    logger.info(
        "             below threshold: %d  (%.1f%%)  |  above: %d  (%.1f%%)",
        below,
        100 * below / len(scores),
        above,
        100 * above / len(scores),
    )


def main() -> None:
    args = parse_args()

    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            f"Train first: python -m training.train --part-type {args.part_type}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_anomaly_model(part_type=args.part_type, force_reload=True)
    extractor = FeatureExtractor(device=device)

    logger.info("=" * 60)
    logger.info("Anomaly Detection Evaluation — %s", args.part_type)
    logger.info("Model: %s", args.model_path)
    logger.info("Threshold: %.4f", detector.threshold)
    logger.info("Memory bank size: %d", len(detector.memory_bank))
    logger.info("=" * 60)

    ok_embeddings = _score_folder(
        extractor, args.data_dir, args.batch_size, args.num_workers
    )
    ok_scores = np.array(
        [detector.score_embedding(emb) for emb in ok_embeddings], dtype=np.float32
    )
    logger.info("OK image score distribution:")
    _print_score_stats(ok_scores, "OK", detector.threshold)

    if args.not_ok_dir is not None:
        logger.info("NOT_OK image score distribution (evaluation only):")
        try:
            nok_embeddings = _score_folder(
                extractor, args.not_ok_dir, args.batch_size, args.num_workers
            )
            nok_scores = np.array(
                [detector.score_embedding(emb) for emb in nok_embeddings],
                dtype=np.float32,
            )
            _print_score_stats(nok_scores, "NOT_OK", detector.threshold)
            gap = float(nok_scores.mean() - ok_scores.mean())
            logger.info("Score gap (mean NOT_OK − mean OK): %.4f", gap)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Could not evaluate NOT_OK folder: %s", exc)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
