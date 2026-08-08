"""Content and perceptual hashing without executable metadata."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a compact difference hash for near-duplicate discovery."""
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels: list[int] = np.asarray(grayscale, dtype=np.uint8).reshape(-1).tolist()
    bits: list[bool] = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        bits.extend(pixels[offset + column] > pixels[offset + column + 1] for column in range(hash_size))
    value = sum(int(bit) << position for position, bit in enumerate(bits))
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    """Count differing bits in equal-width hexadecimal hashes."""
    if len(left) != len(right):
        raise ValueError("Perceptual hashes must have the same width.")
    return (int(left, 16) ^ int(right, 16)).bit_count()
