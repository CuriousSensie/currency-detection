"""Registered EfficientNet-B0 denomination-classification workflows."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pkrvision.constants import DENOMINATIONS, PROJECT_SEED
from pkrvision.research.metrics import bootstrap_intervals, classification_metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return np.asarray(values / values.sum(axis=1, keepdims=True), dtype=np.float64)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit one positive temperature on validation logits only."""
    if logits.ndim != 2 or len(logits) != len(targets) or len(targets) == 0:
        raise ValueError("Validation logits and targets must be non-empty and aligned.")
    candidates = np.geomspace(0.05, 10.0, 2000)
    losses = []
    for temperature in candidates:
        probabilities = _softmax(logits / temperature)
        losses.append(float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean()))
    return float(candidates[int(np.argmin(losses))])


def calibrated_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to calibrated probabilities using a fitted temperature."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    return _softmax(logits / temperature)


def _registered_image_folder(root: Path, transform: Any) -> Any:
    from torchvision import datasets

    class RegisteredImageFolder(datasets.ImageFolder):  # type: ignore[misc]
        def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
            classes = [str(value) for value in DENOMINATIONS]
            available = {path.name for path in Path(directory).iterdir() if path.is_dir()}
            if available != set(classes):
                raise ValueError(f"Expected denomination directories {classes}; found {sorted(available)}.")
            return classes, {name: index for index, name in enumerate(classes)}

    return RegisteredImageFolder(root, transform=transform)


def _transforms(config: dict[str, Any], augmentation: bool) -> tuple[Any, Any]:
    import torch
    from torchvision import transforms

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])
    if not augmentation:
        return evaluation, evaluation
    recipe = config["augmentation"]

    class GaussianNoise:
        def __call__(self, tensor: Any) -> Any:
            return torch.clamp(tensor + torch.randn_like(tensor) * 0.025, 0, 1)

    training = transforms.Compose(
        [
            transforms.RandomPerspective(distortion_scale=float(recipe["perspective_scale"]), p=0.25),
            transforms.RandomRotation(
                float(recipe["rotation_degrees"]), fill=tuple(round(v * 255) for v in (0.485, 0.456, 0.406))
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2)],
                p=float(recipe["brightness_contrast_probability"]),
            ),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=float(recipe["blur_probability"])),
            transforms.ToTensor(),
            transforms.RandomApply([GaussianNoise()], p=float(recipe["noise_probability"])),
            transforms.RandomErasing(p=float(recipe["coarse_dropout_probability"]), scale=(0.02, 0.08)),
            normalize,
        ]
    )
    return training, evaluation


def _evaluate(model: Any, loader: Any, criterion: Any, device: Any) -> tuple[float, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    losses: list[float] = []
    targets: list[int] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            output = model(inputs.to(device))
            losses.append(float(criterion(output, labels.to(device)).item()))
            targets.extend(labels.numpy().tolist())
            logits.append(output.cpu().numpy())
    return float(np.mean(losses)), np.asarray(targets), np.concatenate(logits)


def train_classifier(
    config_path: Path,
    data_root: Path,
    plan_path: Path,
    samples_per_class: int | str,
    augmentation: bool,
    output_dir: Path,
) -> Path:
    """Train one registered ablation condition and emit calibrated predictions."""
    try:
        import timm
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", PROJECT_SEED))
    _seed_everything(seed)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected_plan = next((item for item in plan if item["requested_per_class"] == samples_per_class), None)
    if not selected_plan or selected_plan["status"] != "planned":
        raise ValueError(f"Ablation level {samples_per_class} is not feasible in {plan_path}.")
    selected_stems = set(selected_plan["sample_ids"])
    train_transform, evaluation_transform = _transforms(config, augmentation)
    train_dataset = _registered_image_folder(data_root / "train", train_transform)
    indices = [index for index, (path, _) in enumerate(train_dataset.samples) if Path(path).stem in selected_stems]
    expected = len(selected_stems)
    if len(indices) != expected:
        raise ValueError(f"Expected {expected} registered images but matched {len(indices)}.")
    validation_dataset = _registered_image_folder(data_root / "validation", evaluation_transform)
    test_dataset = _registered_image_folder(data_root / "test", evaluation_transform)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["batch_size"])
    loaders = {
        "train": DataLoader(
            Subset(train_dataset, indices), batch_size=batch_size, shuffle=True, generator=generator, num_workers=2
        ),
        "validation": DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(DENOMINATIONS)).to(device)
    pretrained_config = {
        key: str(model.pretrained_cfg[key])
        for key in ("architecture", "tag", "url", "hf_hub_id")
        if model.pretrained_cfg.get(key)
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best_f1, stale_epochs = -1.0, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(int(config["max_epochs"])):
        model.train()
        for inputs, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        validation_loss, targets, logits = _evaluate(model, loaders["validation"], criterion, device)
        metrics = classification_metrics(
            targets, calibrated_probabilities(logits, 1.0), list(DENOMINATIONS)
        )
        history.append(
            {"epoch": float(epoch + 1), "validation_loss": validation_loss, "validation_macro_f1": metrics.macro_f1}
        )
        if metrics.macro_f1 > best_f1:
            best_f1, stale_epochs = metrics.macro_f1, 0
            torch.save({"state_dict": model.state_dict(), "class_to_idx": train_dataset.class_to_idx}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    _, validation_targets, validation_logits = _evaluate(model, loaders["validation"], criterion, device)
    temperature = fit_temperature(validation_logits, validation_targets)
    validation_probabilities = calibrated_probabilities(validation_logits, temperature)
    _, test_targets, test_logits = _evaluate(model, loaders["test"], criterion, device)
    test_probabilities = calibrated_probabilities(test_logits, temperature)
    validation_metrics = classification_metrics(validation_targets, validation_probabilities, list(DENOMINATIONS))
    test_metrics = classification_metrics(test_targets, test_probabilities, list(DENOMINATIONS))
    state["temperature"] = temperature
    torch.save(state, checkpoint)
    np.savez_compressed(
        output_dir / "validation-predictions.npz",
        targets=validation_targets,
        logits=validation_logits,
        probabilities=validation_probabilities,
    )
    np.savez_compressed(
        output_dir / "test-predictions.npz", targets=test_targets, logits=test_logits, probabilities=test_probabilities
    )
    payload = {
        "status": "complete",
        "samples_per_class": samples_per_class,
        "augmentation": augmentation,
        "seed": seed,
        "subset_sha256": selected_plan["subset_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "temperature": temperature,
        "duration_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
        },
        "hardware": {
            "device": str(device),
            "accelerator": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        },
        "initialization": {"source": "timm_pretrained", **pretrained_config},
        "history": history,
        "validation_metrics": validation_metrics.as_dict(),
        "test_metrics": test_metrics.as_dict(),
        "test_bootstrap_95": bootstrap_intervals(
            test_targets, test_probabilities, iterations=int(config["bootstrap_iterations"])
        ),
        "interval_limitation": "Fixed-test sampling uncertainty only; training-seed variance is excluded.",
    }
    result_path = output_dir / "run.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result_path


def select_classifier(no_augmentation: Path, augmentation: Path, output: Path) -> Path:
    """Select a full-data condition using only registered validation metrics."""
    candidates = [("no_augmentation", no_augmentation), ("registered_augmentation", augmentation)]
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in candidates}
    for name, record in records.items():
        if record.get("samples_per_class") != "full" or record.get("status") != "complete":
            raise ValueError(f"{name} is not a completed full-data run.")
    noaug_f1 = float(records["no_augmentation"]["validation_metrics"]["macro_f1"])
    aug_f1 = float(records["registered_augmentation"]["validation_metrics"]["macro_f1"])
    if abs(noaug_f1 - aug_f1) >= 0.002:
        selected = "no_augmentation" if noaug_f1 > aug_f1 else "registered_augmentation"
        rule = "higher_validation_macro_f1"
    else:
        noaug_ece = float(records["no_augmentation"]["validation_metrics"]["expected_calibration_error"])
        aug_ece = float(records["registered_augmentation"]["validation_metrics"]["expected_calibration_error"])
        selected = "no_augmentation" if noaug_ece <= aug_ece else "registered_augmentation"
        rule = "macro_f1_within_0.002_then_lower_validation_ece"
    payload = {"status": "complete", "selected": selected, "rule": rule, "candidates": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def export_classifier(checkpoint: Path, output: Path) -> Path:
    """Export a trained classifier with a dynamic batch dimension."""
    try:
        import timm
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(DENOMINATIONS))
    model.load_state_dict(state["state_dict"])
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    return output


def validate_onnx_classifier(
    checkpoint: Path,
    model_path: Path,
    test_root: Path,
    output: Path,
    batch_size: int = 64,
    max_probability_drift: float = 1e-4,
) -> Path:
    """Prove ONNX/PyTorch top-1 agreement on the fixed internal test set."""
    try:
        import onnxruntime as ort
        import timm
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    _, evaluation_transform = _transforms({"augmentation": {}}, False)
    dataset = _registered_image_folder(test_root, evaluation_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_class_map = {str(value): index for index, value in enumerate(DENOMINATIONS)}
    if state.get("class_to_idx") != expected_class_map:
        raise ValueError("Checkpoint class order does not match the registered denomination order.")
    reference = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(DENOMINATIONS))
    reference.load_state_dict(state["state_dict"])
    reference.eval()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    temperature = float(state["temperature"])
    compared = 0
    mismatches = 0
    max_logit_drift = 0.0
    max_probability_drift_observed = 0.0
    with torch.no_grad():
        for inputs, _ in loader:
            reference_logits = reference(inputs).numpy()
            onnx_logits = np.asarray(session.run(None, {input_name: inputs.numpy()})[0])
            reference_probabilities = calibrated_probabilities(reference_logits, temperature)
            onnx_probabilities = calibrated_probabilities(onnx_logits, temperature)
            compared += len(inputs)
            mismatches += int(
                np.count_nonzero(reference_probabilities.argmax(axis=1) != onnx_probabilities.argmax(axis=1))
            )
            max_logit_drift = max(max_logit_drift, float(np.max(np.abs(reference_logits - onnx_logits))))
            max_probability_drift_observed = max(
                max_probability_drift_observed,
                float(np.max(np.abs(reference_probabilities - onnx_probabilities))),
            )
    passed = compared == len(dataset) and mismatches == 0 and max_probability_drift_observed <= max_probability_drift
    payload = {
        "status": "passed" if passed else "failed",
        "sample_count": compared,
        "class_order": list(DENOMINATIONS),
        "top1_mismatches": mismatches,
        "top1_agreement": 1.0 - (mismatches / compared),
        "max_absolute_logit_drift": max_logit_drift,
        "max_absolute_probability_drift": max_probability_drift_observed,
        "maximum_allowed_probability_drift": max_probability_drift,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "classifier_onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "runtime": "onnxruntime-cpu",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError(f"ONNX validation failed; inspect {output}.")
    return output


def evaluate_external(checkpoint: Path, external_root: Path, output: Path, batch_size: int = 64) -> Path:
    """Evaluate the selected checkpoint on isolated Abduls crops without fitting."""
    try:
        import timm
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies: pip install -e '.[train]'.") from exc
    _, evaluation_transform = _transforms({"augmentation": {}}, False)
    dataset = _registered_image_folder(external_root, evaluation_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(DENOMINATIONS))
    model.load_state_dict(state["state_dict"])
    _, targets, logits = _evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    probabilities = calibrated_probabilities(logits, float(state["temperature"]))
    metrics = classification_metrics(targets, probabilities, list(DENOMINATIONS))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.parent / "external-predictions.npz",
        targets=targets,
        logits=logits,
        probabilities=probabilities,
    )
    payload = {
        "status": "complete",
        "partition": "cross_dataset_external",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "sample_count": len(targets),
        "metrics": metrics.as_dict(),
        "bootstrap_95": bootstrap_intervals(targets, probabilities),
        "limitation": "Public cross-dataset evaluation; not user-collected real-world validation.",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output
