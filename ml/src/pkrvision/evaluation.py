"""Model evaluation, Pareto selection, and CPU benchmarking."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pkrvision.inference import OnnxClassifier, OnnxDetector
from pkrvision.preprocessing import classifier_tensor


def evaluate_detector(checkpoint: Path, data_yaml: Path, output: Path) -> Path:
    """Evaluate a detector on the untouched test split and retain native metrics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    started = time.time()
    result = YOLO(str(checkpoint)).val(data=str(data_yaml), split="test", plots=True, save_json=True)
    metrics = {str(key): float(value) for key, value in result.results_dict.items()}
    payload = {
        "status": "complete",
        "kind": "detection_evaluation",
        "checkpoint_sha256": _sha256(checkpoint),
        "data_yaml_sha256": _sha256(data_yaml),
        "duration_seconds": time.time() - started,
        "metrics": metrics,
        "save_dir": str(result.save_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def choose_detector(nano_result: Path, small_result: Path, output: Path) -> Path:
    """Apply the pre-declared one-percentage-point deployment rule."""
    nano = json.loads(nano_result.read_text(encoding="utf-8"))
    small = json.loads(small_result.read_text(encoding="utf-8"))
    key_candidates = ("metrics/mAP50-95(B)", "map50_95", "map_50_95")

    def metric(payload: dict[str, Any]) -> float:
        for key in key_candidates:
            if key in payload.get("metrics", {}):
                return float(payload["metrics"][key])
        raise ValueError(f"No mAP@0.5:0.95 key found; tried {key_candidates}.")

    nano_score, small_score = metric(nano), metric(small)
    selected = "yolov8n" if nano_score >= small_score - 0.01 else "yolov8s"
    payload = {
        "status": "complete",
        "rule": "select nano when within 0.01 absolute mAP@0.5:0.95 of small",
        "nano_map50_95": nano_score,
        "small_map50_95": small_score,
        "selected": selected,
        "inputs": {"nano_sha256": _sha256(nano_result), "small_sha256": _sha256(small_result)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def benchmark_detector(
    model_path: Path,
    image_paths: list[Path],
    output: Path,
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> Path:
    """Record cold/warm direct-ONNX detector latency on named hardware."""
    if not image_paths:
        raise ValueError("At least one benchmark image is required.")
    load_start = time.perf_counter()
    detector = OnnxDetector(model_path)
    cold_load_ms = (time.perf_counter() - load_start) * 1000
    images = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]
    for index in range(warmup):
        detector.predict(images[index % len(images)], 0.35, 0.5)
    timings: list[float] = []
    for index in range(repeats):
        started = time.perf_counter()
        detector.predict(images[index % len(images)], 0.35, 0.5)
        timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    payload = {
        "status": "complete",
        "kind": "onnx_cpu_benchmark",
        "model_sha256": _sha256(model_path),
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "sample_files": [{"path": str(path), "sha256": _sha256(path)} for path in image_paths],
        "warmup_runs": warmup,
        "measured_runs": repeats,
        "cold_load_ms": cold_load_ms,
        "latency_ms": {
            "mean": statistics.fmean(timings),
            "median": statistics.median(timings),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "samples": timings,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def benchmark_classifier(
    model_path: Path,
    temperature: float,
    image_paths: list[Path],
    output: Path,
    *,
    warmup: int = 10,
    repeats: int = 100,
) -> Path:
    """Measure checksum-bound single-image classifier performance on CPU."""
    if not image_paths:
        raise ValueError("At least one benchmark image is required.")
    if temperature <= 0 or warmup < 0 or repeats <= 0:
        raise ValueError("Temperature and measured repeats must be positive; warmup cannot be negative.")
    images: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    load_started = time.perf_counter()
    classifier = OnnxClassifier(model_path, temperature)
    load_ms = (time.perf_counter() - load_started) * 1000

    first_preprocess_started = time.perf_counter()
    first_tensor = classifier_tensor(images[0])
    first_preprocess_ms = (time.perf_counter() - first_preprocess_started) * 1000
    first_inference_started = time.perf_counter()
    classifier.predict_tensor(first_tensor)
    first_inference_ms = (time.perf_counter() - first_inference_started) * 1000
    for index in range(warmup):
        classifier.predict_tensor(classifier_tensor(images[index % len(images)]))

    preprocessing: list[float] = []
    inference: list[float] = []
    total_started = time.perf_counter()
    for index in range(repeats):
        image = images[index % len(images)]
        started = time.perf_counter()
        tensor = classifier_tensor(image)
        prepared = time.perf_counter()
        classifier.predict_tensor(tensor)
        finished = time.perf_counter()
        preprocessing.append((prepared - started) * 1000)
        inference.append((finished - prepared) * 1000)
    elapsed_seconds = time.perf_counter() - total_started
    totals = [left + right for left, right in zip(preprocessing, inference, strict=True)]

    def distribution(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    payload = {
        "status": "complete",
        "kind": "classifier_onnx_cpu_benchmark",
        "model_sha256": _sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "temperature": temperature,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "sample_files": [{"path": str(path), "sha256": _sha256(path)} for path in image_paths],
        "warmup_runs": warmup,
        "measured_runs": repeats,
        "cold_ms": {
            "session_load": load_ms,
            "first_preprocess": first_preprocess_ms,
            "first_inference": first_inference_ms,
            "load_plus_first_request": load_ms + first_preprocess_ms + first_inference_ms,
        },
        "warm_latency_ms": {
            "preprocess": distribution(preprocessing),
            "inference": distribution(inference),
            "combined": distribution(totals),
        },
        "sequential_throughput_images_per_second": repeats / elapsed_seconds,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "Single-process sequential CPU calls; image decoding excluded and source images preloaded.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
