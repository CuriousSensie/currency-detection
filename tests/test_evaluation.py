import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops

from pkrvision.evaluation import choose_detector
from pkrvision.release import build_release_manifest
from pkrvision.research.corruptions import apply_corruption


def test_detector_selection_applies_predeclared_margin(tmp_path: Path) -> None:
    nano = tmp_path / "nano.json"
    small = tmp_path / "small.json"
    output = tmp_path / "selection.json"
    nano.write_text(json.dumps({"metrics": {"metrics/mAP50-95(B)": 0.701}}))
    small.write_text(json.dumps({"metrics": {"metrics/mAP50-95(B)": 0.710}}))
    choose_detector(nano, small, output)
    assert json.loads(output.read_text())["selected"] == "yolov8n"


def test_detector_selection_uses_small_outside_margin(tmp_path: Path) -> None:
    nano = tmp_path / "nano.json"
    small = tmp_path / "small.json"
    output = tmp_path / "selection.json"
    nano.write_text(json.dumps({"metrics": {"map50_95": 0.69}}))
    small.write_text(json.dumps({"metrics": {"map50_95": 0.71}}))
    choose_detector(nano, small, output)
    assert json.loads(output.read_text())["selected"] == "yolov8s"


def test_release_manifest_hashes_every_serving_artifact(tmp_path: Path) -> None:
    (tmp_path / "classifier.onnx").write_bytes(b"classifier")
    (tmp_path / "model-card.md").write_text("# Model card\n")
    internal = tmp_path / "internal.json"
    external = tmp_path / "external.json"
    passing = {
        "macro_f1": 0.95,
        "per_class_recall": {str(index): 0.9 for index in range(7)},
    }
    checkpoint_sha256 = "b" * 64
    internal.write_text(
        json.dumps({"temperature": 1.2, "checkpoint_sha256": checkpoint_sha256, "test_metrics": passing})
    )
    external.write_text(json.dumps({"metrics": {"macro_f1": 0.8}}))
    (tmp_path / "onnx-validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "classifier_onnx_sha256": hashlib.sha256(b"classifier").hexdigest(),
                "checkpoint_sha256": checkpoint_sha256,
                "class_order": [10, 20, 50, 100, 500, 1000, 5000],
            }
        )
    )
    manifest = build_release_manifest(tmp_path, "classifier-v1", "a" * 64, internal, external)
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "complete"
    assert payload["quality_gate"]["passed"] is True
    assert set(payload["artifacts"]) == {"classifier", "model_card", "onnx_validation", "result_summary"}
    assert all(len(item["sha256"]) == 64 for item in payload["artifacts"].values())


def test_synthetic_corruption_is_deterministic_and_non_destructive() -> None:
    source = Image.new("RGB", (80, 40), "white")
    first = apply_corruption(source, "stain", 42)
    second = apply_corruption(source, "stain", 42)
    assert ImageChops.difference(first, second).getbbox() is None
    assert ImageChops.difference(source, first).getbbox() is not None
    assert source.getpixel((0, 0)) == (255, 255, 255)
