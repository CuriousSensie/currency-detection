"""Create the checksum manifest that forms the classification serving boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pkrvision.constants import DENOMINATIONS
from pkrvision.preprocessing import CLASSIFIER_SIZE, IMAGENET_MEAN, IMAGENET_STD, LETTERBOX_RGB

INTERNAL_MACRO_F1_MIN = 0.90
EXTERNAL_MACRO_F1_MIN = 0.70
INTERNAL_PER_CLASS_RECALL_MIN = 0.75


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_release_manifest(
    directory: Path,
    classifier_version: str,
    dataset_manifest_sha256: str,
    internal_result: Path,
    external_result: Path,
) -> Path:
    """Build a complete bundle manifest and evaluate the preregistered quality gate."""
    classifier = directory / "classifier.onnx"
    model_card = directory / "model-card.md"
    onnx_validation = directory / "onnx-validation.json"
    required = {"classifier": classifier, "model_card": model_card, "onnx_validation": onnx_validation}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    internal = json.loads(internal_result.read_text(encoding="utf-8"))
    external = json.loads(external_result.read_text(encoding="utf-8"))
    validation = json.loads(onnx_validation.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("ONNX validation is not marked passed.")
    if validation.get("classifier_onnx_sha256") != _sha256(classifier):
        raise ValueError("ONNX validation refers to a different classifier artifact.")
    if validation.get("checkpoint_sha256") != internal.get("checkpoint_sha256"):
        raise ValueError("ONNX validation and internal result refer to different checkpoints.")
    if validation.get("class_order") != list(DENOMINATIONS):
        raise ValueError("ONNX validation class order is not canonical.")
    temperature = float(internal["temperature"])
    internal_metrics = internal["test_metrics"]
    external_metrics = external["metrics"]
    recalls = {str(key): float(value) for key, value in internal_metrics["per_class_recall"].items()}
    checks = {
        "internal_macro_f1": float(internal_metrics["macro_f1"]) >= INTERNAL_MACRO_F1_MIN,
        "external_macro_f1": float(external_metrics["macro_f1"]) >= EXTERNAL_MACRO_F1_MIN,
        "internal_per_class_recall": all(value >= INTERNAL_PER_CLASS_RECALL_MIN for value in recalls.values()),
    }
    result_summary = directory / "result-summary.json"
    result_summary.write_text(
        json.dumps({"internal": internal_metrics, "external": external_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        key: {"filename": path.name, "sha256": _sha256(path)}
        for key, path in {**required, "result_summary": result_summary}.items()
    }
    payload = {
        "schema_version": 2,
        "status": "complete",
        "task": "single-banknote-image-classification",
        "classifier_version": classifier_version,
        "class_order": list(DENOMINATIONS),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "calibration": {"method": "validation_temperature_scaling", "temperature": temperature},
        "preprocessing": {
            "size": [CLASSIFIER_SIZE, CLASSIFIER_SIZE],
            "resize": "aspect-ratio-preserving-letterbox",
            "letterbox_rgb": list(LETTERBOX_RGB),
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "quality_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "internal_macro_f1": INTERNAL_MACRO_F1_MIN,
                "external_macro_f1": EXTERNAL_MACRO_F1_MIN,
                "internal_per_class_recall": INTERNAL_PER_CLASS_RECALL_MIN,
            },
        },
        "training_data_summary": {"source": "UCP v1", "partition": "internal"},
        "evaluation_summary": {
            "internal": internal_metrics,
            "external": external_metrics,
            "external_boundary": "public cross-dataset evaluation; not real-world validation",
        },
        "artifacts": artifacts,
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest
