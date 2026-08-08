"""Deterministic synthetic corruptions for proxy-only anomaly evaluation."""

from __future__ import annotations

import random
from typing import Literal

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

Corruption = Literal["tear", "stain", "occlusion", "fade", "crease"]


def apply_corruption(image: Image.Image, kind: Corruption, seed: int) -> Image.Image:
    """Apply a recorded synthetic corruption without modifying the source image."""
    rng = random.Random(seed)
    result = image.convert("RGB").copy()
    width, height = result.size
    draw = ImageDraw.Draw(result, "RGBA")
    if kind == "tear":
        edge = rng.choice(["left", "right"])
        span = max(2, width // 8)
        points = [(0 if edge == "left" else width, y) for y in range(0, height + 1, max(1, height // 6))]
        jittered = [(x + rng.randint(0, span) * (1 if edge == "left" else -1), y) for x, y in points]
        polygon = [(0, 0), *jittered, (0, height)] if edge == "left" else [(width, 0), *jittered, (width, height)]
        draw.polygon(polygon, fill=(18, 18, 18, 255))
    elif kind == "stain":
        radius = max(4, min(width, height) // 5)
        cx, cy = rng.randrange(width), rng.randrange(height)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(92, 55, 20, 95))
        result = result.filter(ImageFilter.GaussianBlur(max(1, radius // 7)))
    elif kind == "occlusion":
        box_width, box_height = max(4, width // 3), max(4, height // 4)
        x, y = rng.randrange(max(1, width - box_width)), rng.randrange(max(1, height - box_height))
        draw.rectangle((x, y, x + box_width, y + box_height), fill=(20, 24, 26, 255))
    elif kind == "fade":
        result = ImageEnhance.Contrast(result).enhance(0.35)
        result = ImageEnhance.Color(result).enhance(0.30)
    elif kind == "crease":
        x = rng.randrange(max(1, width // 4), max(2, 3 * width // 4))
        draw.line(
            (x, 0, x + rng.randint(-width // 10, width // 10), height),
            fill=(245, 241, 215, 190),
            width=max(1, width // 50),
        )
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return result
