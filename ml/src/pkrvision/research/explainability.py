"""Post-evaluation Grad-CAM diagnostics for the selected classifier."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pkrvision.constants import DENOMINATIONS
from pkrvision.data.hashing import sha256_file
from pkrvision.research.classifier import _registered_image_folder, _transforms


def _selected_indices(targets: np.ndarray, probabilities: np.ndarray, limit: int) -> list[int]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    errors = np.flatnonzero(predictions != targets)
    if len(errors):
        return [int(index) for index in errors[np.argsort(confidence[errors])[::-1]][:limit]]
    correct = np.flatnonzero(predictions == targets)
    return [int(index) for index in correct[np.argsort(confidence[correct])][:limit]]


def _overlay(image: Image.Image, cam: np.ndarray) -> Image.Image:
    normalized = np.clip(cam, 0, 1)
    heat = np.stack(
        [normalized, np.sqrt(normalized) * 0.72, np.zeros_like(normalized)],
        axis=2,
    )
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    blended = np.clip(base * 0.58 + heat * 0.42, 0, 1)
    return Image.fromarray(np.asarray(blended * 255, dtype=np.uint8))


def generate_gradcam_report(
    checkpoint: Path,
    test_root: Path,
    predictions_path: Path,
    output_dir: Path,
    limit: int = 28,
) -> Path:
    """Render deterministic Grad-CAM examples after evaluation without selecting a model."""
    if limit <= 0:
        raise ValueError("Grad-CAM example limit must be positive.")
    try:
        import timm
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    with np.load(predictions_path, allow_pickle=False) as stored:
        targets = np.asarray(stored["targets"], dtype=np.int64)
        probabilities = np.asarray(stored["probabilities"], dtype=np.float64)
    _, evaluation_transform = _transforms({"augmentation": {}}, False)
    dataset = _registered_image_folder(test_root, evaluation_transform)
    if len(dataset) != len(targets) or probabilities.shape != (len(dataset), len(DENOMINATIONS)):
        raise ValueError("Prediction artifact is not aligned with the fixed test tree.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_class_map = {str(value): index for index, value in enumerate(DENOMINATIONS)}
    if state.get("class_to_idx") != expected_class_map:
        raise ValueError("Checkpoint class order does not match the registered denomination order.")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(DENOMINATIONS))
    model.load_state_dict(state["state_dict"])
    model.eval()
    activation: dict[str, Any] = {}

    def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        activation["value"] = output
        output.retain_grad()

    hook = model.conv_head.register_forward_hook(capture)
    selected = _selected_indices(targets, probabilities, min(limit, len(dataset)))
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, object]] = []
    try:
        for position, index in enumerate(selected):
            tensor, target = dataset[index]
            model.zero_grad(set_to_none=True)
            logits = model(tensor.unsqueeze(0))
            prediction = int(probabilities[index].argmax())
            logits[0, prediction].backward()
            features = activation["value"]
            gradients = features.grad
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam_tensor = torch.relu((weights * features).sum(dim=1))[0]
            maximum = float(cam_tensor.max().item())
            if maximum > 0:
                cam_tensor = cam_tensor / maximum
            cam = np.asarray(cam_tensor.detach().numpy(), dtype=np.float32)
            source_path = Path(dataset.samples[index][0])
            with Image.open(source_path) as source:
                resized_cam = Image.fromarray(np.asarray(cam * 255, dtype=np.uint8)).resize(
                    source.size, Image.Resampling.BILINEAR
                )
                rendered = _overlay(source, np.asarray(resized_cam, dtype=np.float32) / 255.0)
            filename = f"{position:02d}-{source_path.stem}.jpg"
            destination = output_dir / filename
            rendered.save(destination, format="JPEG", quality=92)
            examples.append(
                {
                    "sample_id": source_path.stem,
                    "target_pkr": DENOMINATIONS[int(target)],
                    "predicted_pkr": DENOMINATIONS[prediction],
                    "confidence": float(probabilities[index, prediction]),
                    "correct": prediction == int(target),
                    "image": filename,
                    "image_sha256": sha256_file(destination),
                }
            )
    finally:
        hook.remove()
    predicted = probabilities.argmax(axis=1)
    confusions = Counter(
        f"{DENOMINATIONS[int(target)]}->{DENOMINATIONS[int(prediction)]}"
        for target, prediction in zip(targets, predicted, strict=True)
        if target != prediction
    )
    report = output_dir / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "grad-cam-efficientnet-conv-head",
                "selection_policy": "highest-confidence test errors; lowest-confidence correct cases if no errors",
                "selection_role": "post-evaluation qualitative diagnosis only; never model selection",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "test_sample_count": len(dataset),
                "error_count": int(np.count_nonzero(predicted != targets)),
                "confusion_counts": dict(sorted(confusions.items())),
                "examples": examples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
