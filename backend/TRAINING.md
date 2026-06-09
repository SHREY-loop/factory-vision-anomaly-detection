# Training Guide — Multi-Part Anomaly Detection

## Overview

This system uses **unsupervised anomaly detection**. Only OK (normal) images are required for training. A frozen EfficientNet-B0 backbone extracts 1280-dimensional embeddings; a nearest-neighbour memory bank and calibrated distance threshold classify new images at inference time.

**No NOT_OK images are needed for training.**

Each part type (`part_A`, `part_B`, `part_C`, …) has its own dataset folder and trained model.

---

## Algorithm

1. Load OK images from `datasets/<part_type>/ok/`
2. Extract **1280-dim embeddings** per image using frozen EfficientNet-B0
3. Optionally apply in-memory augmentation (1 original + N augmented views per image)
4. Store embeddings as a **memory bank** (all vectors kept when `CORESET_RATIO=1.0`)
5. Calibrate **decision threshold** at the 95th percentile of leave-one-out nearest-neighbour distances
6. Save `models/<part_type>/anomaly_model.pkl` and `models/<part_type>/training_report.json`

At inference: embedding L2 distance to the nearest memory-bank vector is compared to the threshold. Distance above threshold → `NOT_OK`.

---

## Dataset structure

```
backend/datasets/
└── part_A/
    └── ok/
        ├── part_001.jpg
        └── part_002.jpg
```

See [datasets/README.md](datasets/README.md) for full layout.

---

## Quick start

```powershell
# From backend/
pip install -r requirements.txt

# 1. Place OK images in datasets/part_A/ok/

# 2. Train
python -m training.train --part-type part_A

# 3. Start API
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Repeat for each part type (`part_B`, `part_C`, …).

---

## Training command reference

```powershell
python -m training.train --part-type part_A [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--part-type` | *(required)* | Part identifier, e.g. `part_A` |
| `--data-dir` | `datasets/<part_type>/ok/` | Override OK image folder |
| `--output` | `models/<part_type>/anomaly_model.pkl` | Model output path |
| `--report` | `models/<part_type>/training_report.json` | Training diagnostics |
| `--threshold-percentile` | `95` | Decision threshold percentile (0–100) |
| `--coreset-ratio` | `1.0` | Memory bank size ratio (1.0 = keep all) |
| `--batch-size` | `16` | Feature extraction batch size |
| `--num-workers` | `0` | DataLoader workers (0 recommended on Windows) |
| `--seed` | `42` | Random seed |
| `--augment-copies` | `1` | Augmented views per source image |
| `--no-augment` | off | Disable augmentation |

---

## Tuning threshold

Re-train with a different percentile to adjust sensitivity:

- **Higher (e.g. 98)** — fewer false positives, may miss subtle defects
- **Lower (e.g. 90)** — more sensitive, may flag more OK parts as NOT_OK

```powershell
python -m training.train --part-type part_A --threshold-percentile 90
```

---

## Evaluation

### Console statistics

```powershell
python -m training.evaluate --part-type part_A
```

With optional NOT_OK folder:

```powershell
python -m training.evaluate --part-type part_A --not-ok-dir datasets/part_A/not_ok
```

### Full folder evaluation (CSV, metrics, histogram)

```powershell
python -m tools.evaluate_folder --folder datasets/part_A/ok --label ok --part-type part_A
python -m tools.evaluate_folder --folder datasets/part_A/not_ok --label not_ok --part-type part_A
```

Reports are written to `reports/<part_type>/` (gitignored).

---

## Output artifacts

| File | Description |
|------|-------------|
| `models/<part_type>/anomaly_model.pkl` | Memory bank + threshold (`AnomalyDetector`) |
| `models/<part_type>/training_report.json` | Image count, embeddings, threshold, timing |

These files are gitignored. Copy or archive them after training for deployment.

---

## Prediction history

Every API inference saves audit files under `predictions/<part_type>/`:

```
predictions/part_A/
├── 2026-06-04_10-32-15-123.jpg
└── 2026-06-04_10-32-15-123.json
```

---

## Configuration

Hyperparameters live in `training/config.py`:

- `CORESET_RATIO = 1.0` — keep all embeddings (best for small datasets)
- `THRESHOLD_PERCENTILE = 95.0` — decision boundary calibration
- `AUGMENT_ENABLED = True` — in-memory training augmentation

---

## Roadmap

| Phase | Feature |
|-------|---------|
| ✅ Phase 1 | EfficientNet-B0 anomaly detection (current) |
| 🔲 Phase 2 | YOLO defect localisation + bounding boxes |
| 🔲 Phase 3 | Grad-CAM anomaly heatmaps |
| 🔲 Phase 4 | Live camera inspection stream |
