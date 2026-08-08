"""Shared image preprocessing for classification training and serving."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

CLASSIFIER_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LETTERBOX_RGB = tuple(round(value * 255) for value in IMAGENET_MEAN)


def letterbox(image: Image.Image, size: int = CLASSIFIER_SIZE) -> Image.Image:
    """Normalize orientation and fit an RGB image without changing its aspect ratio."""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    contained = ImageOps.contain(normalized, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), LETTERBOX_RGB)
    canvas.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return canvas


def classifier_tensor(image: Image.Image) -> np.ndarray:
    """Return one NCHW ImageNet-normalized float32 classification tensor."""
    value = np.asarray(letterbox(image), dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return ((value - mean) / std)[None]
