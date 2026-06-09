# Factory Vision Inspection System

Industrial quality inspection: upload a manufactured part image, select the part type, and receive **OK** / **NOT OK** with confidence and anomaly score.

Uses **unsupervised anomaly detection** — train on OK images only; no defect examples required.

## Architecture

```
React UI  →  POST /predict?part_type=part_A  →  FastAPI
    →  EfficientNet-B0 (frozen)  →  1280-dim embedding
    →  Memory bank nearest-neighbour  →  JSON verdict
```

One trained model per part type (`part_A`, `part_B`, `part_C`, …).

## Project structure

```
├── frontend/                 # React + TypeScript + Vite
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       └── types/
├── backend/
│   ├── main.py               # FastAPI entry point
│   ├── routes/               # /predict, /parts, /health
│   ├── services/             # Inference orchestration
│   ├── model/                # FeatureExtractor + AnomalyDetector
│   ├── schemas/              # Pydantic response models
│   ├── training/             # train.py, evaluate.py, dataset, config
│   ├── tools/                # evaluate_folder.py (batch evaluation)
│   ├── datasets/             # Per-part OK images (gitignored)
│   ├── models/               # Per-part .pkl models (gitignored)
│   ├── predictions/          # Runtime inference audit (gitignored)
│   └── reports/              # Evaluation outputs (gitignored)
└── README.md
```

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+

## Backend setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Frontend setup

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

App: http://localhost:5173 — Vite proxies `/predict`, `/parts`, and `/health` to port 8000.

## Train a model

1. Place OK images in `backend/datasets/part_A/ok/` (see [backend/datasets/README.md](backend/datasets/README.md)).
2. Train:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m training.train --part-type part_A
```

3. Restart uvicorn if already running. The UI loads available parts from `GET /parts`.

Full guide: **[backend/TRAINING.md](backend/TRAINING.md)**

## Use the UI

1. Start backend and frontend.
2. Open http://localhost:5173.
3. Select a **Part Type** from the dropdown.
4. Upload a JPG/PNG image.
5. Click **Analyze Part**.

## API

### `GET /parts`

```json
{ "parts": ["part_A", "part_B", "part_C"] }
```

### `GET /health`

```json
{
  "status": "healthy",
  "available_parts": ["part_A", "part_B"],
  "loaded_parts": ["part_A"]
}
```

### `POST /predict?part_type=part_A`

Multipart form field: `file` (jpg, jpeg, or png; max 10 MB).

```powershell
curl -X POST "http://127.0.0.1:8000/predict?part_type=part_A" -F "file=@C:\path\to\part.jpg"
```

Example response:

```json
{
  "status": "OK",
  "confidence": 96.4,
  "anomaly_score": 0.12,
  "raw_distance": 1.85,
  "threshold": 4.01,
  "part_type": "part_A",
  "defect_type": null,
  "bounding_boxes": [],
  "heatmap": null
}
```

### Error responses

| Status | Cause |
|--------|-------|
| 400 | Missing/invalid `part_type`, bad file type, empty file, file too large, undecodable image |
| 404 | No trained model for the requested part type |
| 500 | Unexpected server error |

## Evaluation

```powershell
# Console statistics
python -m training.evaluate --part-type part_A

# Full reports (CSV, metrics, histogram)
python -m tools.evaluate_folder --folder datasets/part_A/ok --label ok --part-type part_A
```

## Known limitations

- One model per part type; part type must be specified for every prediction.
- No defect localisation (bounding boxes) or heatmaps yet.
- No authentication, camera streaming, or cloud deployment in this release.
- Training images and model binaries are not committed to git — reproduce locally after clone.

## Phase 1 scope

Included: multi-part image upload, unsupervised training, memory-bank inference, result display.

Not included: cameras, live video, AWS, auth, dashboards, databases, YOLO, Grad-CAM.
