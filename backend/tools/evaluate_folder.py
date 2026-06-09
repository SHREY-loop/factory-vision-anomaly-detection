"""
Folder-level evaluation utility for Phase 1 validation.

Runs inference on every image in a given folder using the trained anomaly
detection model and writes structured CSV, metrics, diagnostic, and histogram
reports.  Does not modify scoring, threshold, or model logic.

Usage (from backend/ directory)
---------------------------------
    python -m tools.evaluate_folder --folder datasets/part_B/ok --label ok --part-type part_B
    python -m tools.evaluate_folder --folder datasets/part_C/not_ok --label not_ok --part-type part_C

Output (under reports/<part_type>/)
-----------------------------------
    <timestamp>_<folder>_results.csv       — per-image scores, sorted by raw_distance desc
    <timestamp>_<folder>_evaluation.txt    — accuracy, precision, recall, F1, confusion matrix
    <timestamp>_<folder>_diagnostic.txt   — top-10 highest-distance ground-truth OK images
    distance_histogram.png                 — distance distribution with threshold line
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path when run as a script
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from model.efficientnet import FeatureExtractor, load_anomaly_model  # noqa: E402
from training.config import (  # noqa: E402
    DEFAULT_REPORTS_DIR,
    get_part_model_path,
    validate_part_type,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)

REPORTS_ROOT: Path = DEFAULT_REPORTS_DIR

_SELF_MATCH_EPSILON = 1e-4

_CSV_FIELDNAMES = [
    "filename",
    "prediction",
    "confidence",
    "raw_distance",
    "loo_raw_distance",
    "loo_prediction",
    "threshold",
    "anomaly_score",
]


@dataclass(frozen=True)
class ClassificationMetrics:
    total: int
    predicted_ok: int
    predicted_not_ok: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run anomaly detection inference on a folder of images and "
            "save CSV, metrics, diagnostic, and histogram reports."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Path to the folder containing images to evaluate.",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["ok", "not_ok", "unknown"],
        default="unknown",
        help=(
            "Expected ground-truth label for images in this folder. "
            "Use 'ok' for normal images, 'not_ok' for defective images."
        ),
    )
    parser.add_argument(
        "--part-type",
        type=str,
        required=True,
        help="Part type (e.g. part_A, part_B, part_C). Resolves model path and report directory.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to anomaly_model.pkl. Defaults to models/<part-type>/anomaly_model.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report output directory. Defaults to reports/<part-type>/ when --part-type is set.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image console output. Only show summary.",
    )
    args = parser.parse_args()
    args.part_type = validate_part_type(args.part_type)
    return args


def _resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path
    return get_part_model_path(args.part_type)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return REPORTS_ROOT / args.part_type


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------


def collect_images(folder: Path) -> list[Path]:
    """Return sorted list of image files in *folder* (non-recursive)."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not images:
        raise ValueError(
            f"No supported images found in: {folder}\n"
            f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    return images


# ---------------------------------------------------------------------------
# Inference on a single image
# ---------------------------------------------------------------------------


def _compute_loo_distance(
    detector,
    embedding: np.ndarray,
) -> float:
    """
    Nearest-neighbour distance excluding near-exact self matches.

    Mirrors training threshold calibration (_compute_training_distances) so
    evaluation on the training folder reflects real inference on new captures.
    """
    diff = detector.memory_bank - embedding[np.newaxis, :]
    distances = np.linalg.norm(diff, axis=1)
    other = distances[distances > _SELF_MATCH_EPSILON]
    if len(other) == 0:
        return float(distances.min())
    return float(other.min())


def _infer_single(
    image_path: Path,
    extractor: FeatureExtractor,
    detector,
) -> dict:
    """Run inference on one image file. Returns scored fields incl. LOO distance."""
    raw_bytes = image_path.read_bytes()
    pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    embedding = extractor.extract(pil_image)
    result = detector.predict(embedding)
    loo_raw = _compute_loo_distance(detector, embedding)
    result["loo_raw_distance"] = round(loo_raw, 6)
    result["loo_prediction"] = "NOT_OK" if loo_raw > detector.threshold else "OK"
    result["image_shape"] = f"{pil_image.width}x{pil_image.height}"
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _expected_status(label: str) -> str | None:
    if label == "ok":
        return "OK"
    if label == "not_ok":
        return "NOT_OK"
    return None


def _compute_metrics(rows: list[dict], expected_label: str) -> ClassificationMetrics:
    total = len(rows)
    predicted_ok = sum(1 for r in rows if r["prediction"] == "OK")
    predicted_not_ok = total - predicted_ok

    gt_status = _expected_status(expected_label)
    if gt_status is None:
        return ClassificationMetrics(
            total=total,
            predicted_ok=predicted_ok,
            predicted_not_ok=predicted_not_ok,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
        )

    positive_class = "NOT_OK"
    negative_class = "OK"

    tp = sum(
        1 for r in rows
        if r["expected_label"] == positive_class and r["prediction"] == positive_class
    )
    tn = sum(
        1 for r in rows
        if r["expected_label"] == negative_class and r["prediction"] == negative_class
    )
    fp = sum(
        1 for r in rows
        if r["expected_label"] == negative_class and r["prediction"] == positive_class
    )
    fn = sum(
        1 for r in rows
        if r["expected_label"] == positive_class and r["prediction"] == negative_class
    )

    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return ClassificationMetrics(
        total=total,
        predicted_ok=predicted_ok,
        predicted_not_ok=predicted_not_ok,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _format_rate(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * value:.2f}%"


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _write_csv(report_path: Path, rows: list[dict]) -> None:
    """Write inference results sorted by loo_raw_distance descending."""
    sorted_rows = sorted(
        rows, key=lambda r: float(r["loo_raw_distance"]), reverse=True
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)
    logger.info("CSV report saved → %s", report_path)


def _write_evaluation_report(
    report_path: Path,
    folder: Path,
    expected_label: str,
    rows: list[dict],
    metrics: ClassificationMetrics,
    model_path: Path,
    part_type: str | None,
    elapsed: float,
) -> None:
    """Write full classification metrics and confusion matrix."""
    threshold = float(rows[0]["threshold"]) if rows else 0.0
    dists = [float(r["raw_distance"]) for r in rows]
    loo_dists = [float(r["loo_raw_distance"]) for r in rows]
    scores = [float(r["anomaly_score"]) for r in rows]
    loo_failures = [r for r in rows if r["loo_prediction"] == "NOT_OK"]

    false_positives = [
        r for r in rows
        if r["expected_label"] == "OK" and r["prediction"] == "NOT_OK"
    ]
    false_negatives = [
        r for r in rows
        if r["expected_label"] == "NOT_OK" and r["prediction"] == "OK"
    ]

    lines = [
        "=" * 64,
        "  ANOMALY DETECTION — FOLDER EVALUATION REPORT",
        "=" * 64,
        f"  Timestamp      : {datetime.now().isoformat(timespec='seconds')}",
        f"  Folder         : {folder.resolve()}",
        f"  Expected Label : {expected_label.upper()}",
        f"  Part Type      : {part_type}",
        f"  Model          : {model_path.resolve()}",
        f"  Threshold      : {threshold:.6f}",
        f"  Elapsed        : {elapsed:.2f}s",
        "-" * 64,
        "  PREDICTION SUMMARY",
        f"  Total images       : {metrics.total}",
        f"  OK predictions     : {metrics.predicted_ok}",
        f"  NOT OK predictions : {metrics.predicted_not_ok}",
        "-" * 64,
    ]

    if metrics.accuracy is not None:
        lines += [
            "  CLASSIFICATION METRICS",
            f"  Accuracy           : {_format_rate(metrics.accuracy)}",
            f"  Precision          : {_format_rate(metrics.precision)}",
            f"  Recall             : {_format_rate(metrics.recall)}",
            f"  F1 score           : {_format_float(metrics.f1)}",
            "-" * 64,
            "  CONFUSION MATRIX  (positive class = NOT_OK)",
            "                        Predicted",
            "                        OK        NOT_OK",
            f"  Actual OK             {metrics.true_negatives:>6}    {metrics.false_positives:>6}",
            f"  Actual NOT_OK         {metrics.false_negatives:>6}    {metrics.true_positives:>6}",
            "-" * 64,
            f"  False positives    : {metrics.false_positives}",
            f"  False negatives    : {metrics.false_negatives}",
        ]
    else:
        lines += [
            "  CLASSIFICATION METRICS",
            "  (skipped — expected label is 'unknown')",
        ]

    if dists:
        lines += [
            "-" * 64,
            "  RAW DISTANCE (includes self-match — optimistic on training set)",
            f"  Min                : {min(dists):.6f}",
            f"  Max                : {max(dists):.6f}",
            f"  Mean               : {statistics.mean(dists):.6f}",
            f"  Above threshold    : {sum(1 for d in dists if d > threshold)}",
            "-" * 64,
            "  LOO RAW DISTANCE (excludes self-match — matches training calibration)",
            f"  Min                : {min(loo_dists):.6f}",
            f"  Max                : {max(loo_dists):.6f}",
            f"  Mean               : {statistics.mean(loo_dists):.6f}",
            f"  Median             : {statistics.median(loo_dists):.6f}",
            f"  P95                : {float(np.percentile(loo_dists, 95)):.6f}",
        ]
        if len(loo_dists) > 1:
            lines.append(f"  Stdev              : {statistics.stdev(loo_dists):.6f}")
        lines += [
            f"  Below threshold    : {sum(1 for d in loo_dists if d <= threshold)}",
            f"  Above threshold    : {sum(1 for d in loo_dists if d > threshold)}",
            f"  LOO NOT_OK count   : {len(loo_failures)}",
            "-" * 64,
            "  ANOMALY SCORE STATISTICS (self-match path)",
            f"  Min                : {min(scores):.6f}",
            f"  Max                : {max(scores):.6f}",
            f"  Mean               : {statistics.mean(scores):.6f}",
            "=" * 64,
        ]

    if loo_failures:
        lines += [
            "",
            "  LOO FALSE POSITIVES (ground-truth OK, LOO distance above threshold):",
        ]
        for r in sorted(loo_failures, key=lambda x: float(x["loo_raw_distance"]), reverse=True):
            lines.append(
                f"    {r['filename']:<50}  "
                f"loo_dist={r['loo_raw_distance']}  threshold={r['threshold']}"
            )

    if false_positives:
        lines += ["", "  FALSE POSITIVES (ground-truth OK, predicted NOT_OK):"]
        for r in sorted(false_positives, key=lambda x: float(x["raw_distance"]), reverse=True):
            lines.append(
                f"    {r['filename']:<50}  "
                f"distance={r['raw_distance']}  threshold={r['threshold']}"
            )

    if false_negatives:
        lines += ["", "  FALSE NEGATIVES (ground-truth NOT_OK, predicted OK):"]
        for r in sorted(false_negatives, key=lambda x: float(x["raw_distance"])):
            lines.append(
                f"    {r['filename']:<50}  "
                f"distance={r['raw_distance']}  threshold={r['threshold']}"
            )

    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    logger.info("Evaluation report saved → %s", report_path)
    print("\n" + text)


def _write_diagnostic_report(
    report_path: Path,
    rows: list[dict],
    expected_label: str,
    top_n: int = 10,
) -> None:
    """
    List the top-N highest-distance ground-truth OK images.

    These are the borderline or misclassified OK samples most likely to fail.
    """
    ok_rows = [r for r in rows if r["expected_label"] == "OK"]
    if not ok_rows:
        lines = [
            "=" * 64,
            "  DIAGNOSTIC REPORT — TOP OK SAMPLES BY DISTANCE",
            "=" * 64,
            f"  Expected label for folder: {expected_label.upper()}",
            "",
            "  No ground-truth OK images in this evaluation run.",
            "  Run with --label ok to populate this section.",
            "=" * 64,
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Diagnostic report saved → %s", report_path)
        return

    top = sorted(ok_rows, key=lambda r: float(r["loo_raw_distance"]), reverse=True)[:top_n]
    threshold = float(top[0]["threshold"]) if top else 0.0

    lines = [
        "=" * 64,
        "  DIAGNOSTIC REPORT — TOP OK SAMPLES BY DISTANCE",
        "=" * 64,
        f"  Ground-truth OK images evaluated : {len(ok_rows)}",
        f"  Threshold                        : {threshold:.6f}",
        f"  Showing top {min(top_n, len(top))} highest-distance OK images",
        "-" * 64,
        "",
    ]

    for rank, row in enumerate(top, start=1):
        loo_fail = row["loo_prediction"] == "NOT_OK"
        status_tag = "LOO_FAIL" if loo_fail else "pass"
        lines += [
            f"  #{rank}  {row['filename']}",
            f"       loo_distance  = {row['loo_raw_distance']}",
            f"       self_distance = {row['raw_distance']}",
            f"       threshold     = {row['threshold']}",
            f"       loo_prediction = {row['loo_prediction']}  ({status_tag})",
            f"       self_prediction = {row['prediction']}",
            f"       confidence = {row['confidence']}%",
            f"       anomaly_score = {row['anomaly_score']}",
            "",
        ]

    lines.append("=" * 64)
    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    logger.info("Diagnostic report saved → %s", report_path)


def _write_distance_histogram(
    output_path: Path,
    rows: list[dict],
    part_type: str | None,
    folder_name: str,
) -> None:
    """Plot all image distances with a vertical threshold line."""
    if not rows:
        return

    loo_distances = [float(r["loo_raw_distance"]) for r in rows]
    threshold = float(rows[0]["threshold"])
    part_label = part_type

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        loo_distances,
        bins=min(30, max(10, len(loo_distances) // 2)),
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
    )
    ax.axvline(threshold, color="#C44E52", linestyle="--", linewidth=2, label=f"threshold = {threshold:.4f}")
    ax.set_xlabel("LOO raw nearest-neighbor distance (self-match excluded)")
    ax.set_ylabel("Image count")
    ax.set_title(f"LOO distance distribution — {part_label} ({folder_name}, n={len(loo_distances)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Distance histogram saved → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    model_path = _resolve_model_path(args)
    output_dir = _resolve_output_dir(args)

    if not model_path.is_file():
        logger.error(
            "Trained model not found: %s\nRun training first:\n"
            "  cd backend && python -m training.train --part-type %s",
            model_path,
            args.part_type,
        )
        sys.exit(1)

    images = collect_images(args.folder)
    logger.info("Found %d images in %s", len(images), args.folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    extractor = FeatureExtractor(device=device)
    detector = load_anomaly_model(args.part_type, force_reload=True)

    logger.info(
        "Model loaded — memory bank: %d vectors  threshold: %.6f",
        len(detector.memory_bank),
        detector.threshold,
    )
    logger.info("=" * 56)

    expected_status = _expected_status(args.label)
    rows: list[dict] = []
    errors: list[str] = []
    start_time = time.time()

    for idx, img_path in enumerate(images, start=1):
        try:
            result = _infer_single(img_path, extractor, detector)
            prediction = result["status"]

            row = {
                "filename": img_path.name,
                "prediction": prediction,
                "confidence": result["confidence"],
                "anomaly_score": result["anomaly_score"],
                "raw_distance": result["raw_distance"],
                "loo_raw_distance": result["loo_raw_distance"],
                "loo_prediction": result["loo_prediction"],
                "threshold": result["threshold"],
                "expected_label": expected_status or "UNKNOWN",
            }
            rows.append(row)

            if not args.quiet:
                match = ""
                if expected_status is not None:
                    loo_ok = result["loo_prediction"] == expected_status
                    match = "[LOO_PASS]" if loo_ok else "[LOO_FAIL]"
                logger.info(
                    "[%3d/%d] %-50s  self=%s loo=%s  dist=%.4f loo=%.4f  %s",
                    idx,
                    len(images),
                    img_path.name[:50],
                    prediction,
                    result["loo_prediction"],
                    result["raw_distance"],
                    result["loo_raw_distance"],
                    match,
                )

        except Exception as exc:  # noqa: BLE001
            logger.warning("[%3d/%d] SKIPPED %s — %s", idx, len(images), img_path.name, exc)
            errors.append(img_path.name)

    elapsed = time.time() - start_time
    logger.info("=" * 56)
    logger.info(
        "Inference complete — %d/%d images processed in %.2fs",
        len(rows),
        len(images),
        elapsed,
    )
    if errors:
        logger.warning("%d image(s) skipped due to errors: %s", len(errors), errors)

    if not rows:
        logger.error("No images were successfully processed. Exiting.")
        sys.exit(1)

    metrics = _compute_metrics(rows, args.label)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_slug = args.folder.resolve().name.replace(" ", "_")
    base_name = f"{timestamp}_{folder_slug}"

    csv_path = output_dir / f"{base_name}_results.csv"
    evaluation_path = output_dir / f"{base_name}_evaluation.txt"
    diagnostic_path = output_dir / f"{base_name}_diagnostic.txt"
    histogram_path = output_dir / "distance_histogram.png"

    _write_csv(csv_path, rows)
    _write_evaluation_report(
        evaluation_path,
        args.folder,
        args.label,
        rows,
        metrics,
        model_path,
        args.part_type,
        elapsed,
    )
    _write_diagnostic_report(diagnostic_path, rows, args.label)
    _write_distance_histogram(
        histogram_path,
        rows,
        args.part_type,
        args.folder.name,
    )


if __name__ == "__main__":
    main()
