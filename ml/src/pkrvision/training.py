"""Thin, configuration-driven training entrypoints."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _environment() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or "unavailable",
    }


def train_detector(config_path: Path) -> Path:
    """Train YOLO from a checked-in config and write a provenance record."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(str(config["project"])) / str(config["name"])
    started = time.time()
    model = YOLO(config["model"])
    result = model.train(**{key: value for key, value in config.items() if key != "model"})
    provenance = {
        "status": "complete",
        "kind": "detector_training",
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "environment": _environment(),
        "started_unix": started,
        "duration_seconds": time.time() - started,
        "save_dir": str(result.save_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path


def export_detector(checkpoint: Path, output_dir: Path) -> Path:
    """Export a trained detector to ONNX through Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    exported = Path(YOLO(str(checkpoint)).export(format="onnx", imgsz=640, simplify=True, dynamic=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "detector.onnx"
    destination.write_bytes(exported.read_bytes())
    return destination
