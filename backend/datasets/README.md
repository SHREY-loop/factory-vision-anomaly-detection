# Training dataset layout

Each part type has its own dataset folder. Training uses **OK images only** (unsupervised anomaly detection).

```
backend/datasets/
├── part_A/
│   └── ok/              # Good parts — required for training
│       ├── image_001.jpg
│       └── ...
├── part_B/
│   └── ok/
└── part_C/
    └── ok/
```

Optional evaluation folders (not used during training):

```
backend/datasets/
└── part_A/
    └── not_ok/          # Defective parts — for offline evaluation only
```

## Requirements

- **Formats:** `.jpg`, `.jpeg`, `.png` (training loader also accepts `.bmp`, `.webp`, `.tiff`)
- **Minimum:** At least one OK image per part type (20+ recommended for stable thresholds)
- **Folder names:** `ok` for training images; `not_ok` for optional evaluation

## Training

Place images in `datasets/<part_type>/ok/`, then:

```powershell
cd backend
python -m training.train --part-type part_A
```

## Tips

- Use consistent lighting and viewpoint when possible.
- Train one model per part type — do not mix different parts in the same folder.
- Images stay on your machine; dataset folders are gitignored (only `.gitkeep` placeholders are tracked).
