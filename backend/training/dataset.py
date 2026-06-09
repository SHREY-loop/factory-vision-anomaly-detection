"""
PyTorch dataset utilities for anomaly detection.

Expected layout — OK images in a per-part flat folder:

    datasets/
    └── part_A/
        └── ok/
            ├── image1.jpg
            └── image2.jpg

No class labels, no train/val/test split.
All images are used for memory bank construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from training.augmentation import IndustrialPartAugmentor
from training.config import (
    AUGMENT_COPIES_PER_IMAGE,
    AUGMENT_ENABLED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    INPUT_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)


class OKImageDataset(Dataset):
    """
    Flat-folder dataset containing only OK (normal) factory part images.

    Parameters
    ----------
    data_dir:
        Path to the folder containing OK images directly (no sub-folders).
    transform:
        torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        data_dir: Path | str,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform or build_eval_transform()
        self.image_paths: list[Path] = self._collect_images()
        logger.info(
            "OKImageDataset: found %d images in %s",
            len(self.image_paths),
            self.data_dir,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return (tensor, image_path_str) — path is useful for diagnostics."""
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_images(self) -> list[Path]:
        """Validate directory and collect image file paths."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}\n"
                "Create it and place OK images inside:\n"
                f"  {self.data_dir / 'image1.jpg'}\n"
                f"  {self.data_dir / 'image2.jpg'}\n"
                "  ..."
            )

        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"Expected a directory, got a file: {self.data_dir}"
            )

        paths = sorted(
            p
            for p in self.data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

        if not paths:
            raise ValueError(
                f"No images found in {self.data_dir}\n"
                f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}\n"
                "Training cannot proceed without OK images."
            )

        return paths


class OKImageTrainingDataset(OKImageDataset):
    """
    Training dataset that expands each OK image into original + augmented views.

    Augmented copies are generated in memory; files on disk are never modified.
    Length = num_images × (1 + augment_copies).
    """

    def __init__(
        self,
        data_dir: Path | str,
        augment_copies: int = AUGMENT_COPIES_PER_IMAGE,
        seed: int = DEFAULT_SEED,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__(data_dir=data_dir, transform=transform)
        if augment_copies < 0:
            raise ValueError("augment_copies must be >= 0")
        self.augment_copies = augment_copies
        self._augmentor = IndustrialPartAugmentor(seed=seed)
        self._seed = seed
        self._views_per_image = 1 + augment_copies
        logger.info(
            "OKImageTrainingDataset: %d images × %d views = %d training samples",
            len(self.image_paths),
            self._views_per_image,
            len(self),
        )

    def __len__(self) -> int:
        return len(self.image_paths) * self._views_per_image

    def __getitem__(self, index: int):
        image_idx = index // self._views_per_image
        view_idx = index % self._views_per_image
        path = self.image_paths[image_idx]

        image = Image.open(path).convert("RGB")
        if view_idx > 0:
            variant_seed = self._seed + index
            image = self._augmentor(image, variant_seed=variant_seed)

        tensor = self.transform(image)
        return tensor, str(path)


def build_eval_transform() -> transforms.Compose:
    """
    Deterministic preprocessing pipeline for feature extraction.

    No augmentation — identical transforms at training and inference
    ensure embedding consistency.
    """
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def create_dataloader(
    data_dir: Path | str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    *,
    augment: bool = AUGMENT_ENABLED,
    augment_copies: int = AUGMENT_COPIES_PER_IMAGE,
) -> DataLoader:
    """
    Build a DataLoader over all OK images in *data_dir*.

    When *augment* is True, each source image yields 1 original plus
    *augment_copies* in-memory augmented views before feature extraction.

    Parameters
    ----------
    data_dir:
        Path to the flat OK image folder.
    batch_size:
        Number of images per batch during feature extraction.
    num_workers:
        DataLoader worker processes (0 = main process, recommended on Windows).
    seed:
        Random seed for reproducible augmentation.
    augment:
        Enable in-memory training augmentation.
    augment_copies:
        Number of augmented copies per source image (0 = originals only).

    Returns
    -------
    DataLoader yielding (tensor_batch, path_batch) tuples.
    """
    import torch

    eval_transform = build_eval_transform()
    if augment and augment_copies > 0:
        dataset: OKImageDataset | OKImageTrainingDataset = OKImageTrainingDataset(
            data_dir=data_dir,
            augment_copies=augment_copies,
            seed=seed,
            transform=eval_transform,
        )
    else:
        dataset = OKImageDataset(data_dir=data_dir, transform=eval_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # deterministic order for reproducibility
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
