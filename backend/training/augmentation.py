"""
In-memory training augmentations for industrial part images.

Applied only during training — source files on disk are never modified.
Augmentations are mild and preserve part geometry (no flips, no heavy color shifts).
"""

from __future__ import annotations

import random
from typing import Final

from PIL import Image, ImageEnhance
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

# Hard limits enforced by design (do not exceed via public API).
MAX_ROTATION_DEG: Final[float] = 5.0
BRIGHTNESS_DELTA: Final[float] = 0.15
CONTRAST_DELTA: Final[float] = 0.15
SCALE_MIN: Final[float] = 0.95
SCALE_MAX: Final[float] = 1.05
MAX_TRANSLATE_PX: Final[int] = 10


class IndustrialPartAugmentor:
    """
    Generate a mildly perturbed copy of a PIL image in memory.

    Transforms (all random within bounds):
        - Rotation ±5°
        - Brightness ±15 %
        - Contrast ±15 %
        - Zoom 95–105 %
        - Translation up to ±10 px

    No horizontal/vertical flips. No blur or saturation/hue shifts.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def __call__(
        self,
        image: Image.Image,
        *,
        variant_seed: int | None = None,
    ) -> Image.Image:
        """Return an augmented RGB copy; the input image is not modified."""
        rng = random.Random(variant_seed) if variant_seed is not None else self._rng
        augmented = image.copy()

        angle = rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        scale = rng.uniform(SCALE_MIN, SCALE_MAX)
        tx = rng.randint(-MAX_TRANSLATE_PX, MAX_TRANSLATE_PX)
        ty = rng.randint(-MAX_TRANSLATE_PX, MAX_TRANSLATE_PX)

        fill = _edge_fill_color(augmented)
        augmented = TF.affine(
            augmented,
            angle=angle,
            translate=(tx, ty),
            scale=scale,
            shear=0.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )

        brightness = rng.uniform(1.0 - BRIGHTNESS_DELTA, 1.0 + BRIGHTNESS_DELTA)
        augmented = ImageEnhance.Brightness(augmented).enhance(brightness)

        contrast = rng.uniform(1.0 - CONTRAST_DELTA, 1.0 + CONTRAST_DELTA)
        augmented = ImageEnhance.Contrast(augmented).enhance(contrast)

        return augmented


def _edge_fill_color(image: Image.Image) -> tuple[int, int, int]:
    """Sample a neutral fill from the image border (avoids harsh black borders)."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < 2 or h < 2:
        return (0, 0, 0)
    pixels = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((w - 1, 0)),
        rgb.getpixel((0, h - 1)),
        rgb.getpixel((w - 1, h - 1)),
    ]
    return (
        sum(p[0] for p in pixels) // 4,
        sum(p[1] for p in pixels) // 4,
        sum(p[2] for p in pixels) // 4,
    )


def augmentation_spec() -> dict[str, str | float | int]:
    """Return a serialisable summary of active augmentation parameters."""
    return {
        "rotation_degrees": f"±{MAX_ROTATION_DEG}",
        "brightness": f"±{int(BRIGHTNESS_DELTA * 100)}%",
        "contrast": f"±{int(CONTRAST_DELTA * 100)}%",
        "zoom_scale": f"{SCALE_MIN}–{SCALE_MAX}",
        "max_translation_px": MAX_TRANSLATE_PX,
        "horizontal_flip": False,
        "vertical_flip": False,
    }
