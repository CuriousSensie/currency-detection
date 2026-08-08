"""Embedding export, anomaly fitting, and proxy evaluation workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pkrvision.anomaly import ConditionalKNNIndex
from pkrvision.constants import DENOMINATIONS
from pkrvision.inference import OnnxEmbeddingModel
from pkrvision.research.corruptions import Corruption, apply_corruption


def export_embedder(checkpoint: Path, output: Path) -> Path:
    """Export the registered classifier backbone as a non-executable ONNX feature model."""
    try:
        import timm
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(DENOMINATIONS))
    model.load_state_dict(state["state_dict"])
    model.reset_classifier(0)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        output,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=18,
    )
    return output


def extract_embeddings(model_path: Path, data_root: Path, output: Path, batch_size: int = 64) -> Path:
    """Extract embeddings and non-pickle sample metadata from class directories."""
    model = OnnxEmbeddingModel(model_path)
    items = sorted(
        (path, int(path.parent.name))
        for path in data_root.glob("*/*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not items:
        raise ValueError(f"No class-directory images found below {data_root}.")
    embeddings: list[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        batch = [Image.open(path).convert("RGB") for path, _ in items[start : start + batch_size]]
        embeddings.append(model.embed(batch))
        for image in batch:
            image.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.concatenate(embeddings),
        labels=np.asarray([label for _, label in items], dtype=np.int64),
        sample_ids=np.asarray([path.stem for path, _ in items], dtype="U80"),
    )
    return output


def fit_anomaly_index(train_embeddings: Path, validation_embeddings: Path, output: Path) -> Path:
    """Fit references on train and calibrate percentiles on validation only."""
    train = np.load(train_embeddings, allow_pickle=False)
    validation = np.load(validation_embeddings, allow_pickle=False)
    index = ConditionalKNNIndex.fit(
        train["embeddings"],
        train["labels"],
        validation["embeddings"],
        validation["labels"],
        k=5,
        quantile=0.95,
    )
    missing = sorted(set(DENOMINATIONS) - set(index.references))
    if missing:
        raise ValueError(f"Insufficient train/validation references for denominations: {missing}")
    index.save(output)
    return output


def evaluate_anomaly_proxy(
    embedder_path: Path,
    index_path: Path,
    test_root: Path,
    output_dir: Path,
) -> Path:
    """Evaluate clean versus deterministic synthetic corruptions with pair provenance."""
    embedder = OnnxEmbeddingModel(embedder_path)
    index = ConditionalKNNIndex.load(index_path)
    source_paths = sorted(
        path for path in test_root.glob("*/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not source_paths:
        raise ValueError(f"No test crops found below {test_root}.")
    kinds: tuple[Corruption, ...] = ("tear", "stain", "occlusion", "fade", "crease")
    labels: list[int] = []
    scores: list[float] = []
    pairs: list[dict[str, object]] = []
    derived_root = output_dir / "corruptions"
    for source_position, source_path in enumerate(source_paths):
        denomination = int(source_path.parent.name)
        clean = Image.open(source_path).convert("RGB")
        clean_embedding = embedder.embed([clean])[0]
        clean_score, threshold, clean_flag = index.score(clean_embedding, denomination)
        labels.append(0)
        scores.append(clean_score)
        for kind_position, kind in enumerate(kinds):
            seed = 20260908 + source_position * len(kinds) + kind_position
            damaged = apply_corruption(clean, kind, seed)
            destination = derived_root / str(denomination) / f"{source_path.stem}-{kind}-{seed}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            damaged.save(destination, quality=95)
            damaged_score, _, damaged_flag = index.score(embedder.embed([damaged])[0], denomination)
            labels.append(1)
            scores.append(damaged_score)
            pairs.append(
                {
                    "source": str(source_path),
                    "source_sha256": _sha256(source_path),
                    "derived": str(destination),
                    "derived_sha256": _sha256(destination),
                    "denomination": denomination,
                    "corruption": kind,
                    "seed": seed,
                    "clean_score": clean_score,
                    "clean_flagged": clean_flag,
                    "corrupted_score": damaged_score,
                    "corrupted_flagged": damaged_flag,
                    "threshold": threshold,
                }
            )
        clean.close()
    target = np.asarray(labels)
    values = np.asarray(scores)
    clean_values, damaged_values = values[target == 0], values[target == 1]
    auroc = float(
        np.mean(damaged_values[:, None] > clean_values[None, :])
        + 0.5 * np.mean(damaged_values[:, None] == clean_values[None, :])
    )
    order = np.argsort(values)[::-1]
    sorted_targets = target[order]
    precision = np.cumsum(sorted_targets) / np.arange(1, len(sorted_targets) + 1)
    average_precision = float(np.sum(precision * sorted_targets) / max(1, sorted_targets.sum()))
    payload = {
        "status": "complete",
        "claim_boundary": "synthetic_damage_proxy_only",
        "embedder_sha256": _sha256(embedder_path),
        "index_sha256": _sha256(index_path),
        "clean_samples": len(clean_values),
        "synthetic_samples": len(damaged_values),
        "metrics": {
            "auroc": auroc,
            "average_precision": average_precision,
            "clean_false_positive_rate": float(np.mean(clean_values >= 0.95)),
            "synthetic_recall_at_calibrated_threshold": float(np.mean(damaged_values >= 0.95)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs.jsonl").write_text("".join(json.dumps(pair) + "\n" for pair in pairs), encoding="utf-8")
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
