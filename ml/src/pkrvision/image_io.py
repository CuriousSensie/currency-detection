"""Secure, deterministic image decoding utilities."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageDecodeError(ValueError):
    """Raised when an uploaded image cannot be safely decoded."""


@dataclass(frozen=True)
class DecodedImage:
    """Normalized RGB image and its array representation."""

    image: Image.Image
    array: np.ndarray


def decode_image(payload: bytes, max_pixels: int) -> DecodedImage:
    """Decode bytes as an EXIF-normalized RGB image within a pixel budget."""
    if not payload:
        raise ImageDecodeError("The uploaded file is empty.")
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageDecodeError("Only JPEG, PNG, and WebP images are supported.")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ImageDecodeError(f"Decoded image exceeds the {max_pixels:,}-pixel limit.")
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            normalized.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("The uploaded file is not a valid image.") from exc
    return DecodedImage(image=normalized, array=np.asarray(normalized))
