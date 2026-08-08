"""Backend-independent inference orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from PIL import Image

from pkrvision.anomaly import ConditionalKNNIndex, InsufficientReferenceError
from pkrvision.constants import CLASS_TO_DENOMINATION, DENOMINATIONS
from pkrvision.preprocessing import classifier_tensor


class ModelUnavailableError(RuntimeError):
    """Raised when required, verified model artifacts are absent."""


@dataclass(frozen=True)
class RawDetection:
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float


class AnomalyPayload(TypedDict):
    """Internal typed anomaly payload consumed by the API contract."""

    score: float
    threshold: float
    flagged: bool | None
    label: str


class DetectionPayload(TypedDict):
    """Internal typed detection payload consumed by the API contract."""

    id: str
    bbox_xyxy: tuple[float, float, float, float]
    denomination_pkr: int
    confidence: float
    anomaly: AnomalyPayload


class Detector(Protocol):
    def predict(self, image: np.ndarray, confidence: float, iou: float) -> list[RawDetection]:
        """Detect banknotes in an RGB image."""


class Embedder(Protocol):
    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        """Return one feature embedding per crop."""


class OnnxDetector:
    """Direct ONNX Runtime detector for standard YOLOv8 detection exports."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        height, width = model_input.shape[2:4]
        self._size = (
            int(width) if isinstance(width, int) else 640,
            int(height) if isinstance(height, int) else 640,
        )

    def predict(self, image: np.ndarray, confidence: float, iou: float) -> list[RawDetection]:
        input_width, input_height = self._size
        source_height, source_width = image.shape[:2]
        scale = min(input_width / source_width, input_height / source_height)
        resized_width, resized_height = round(source_width * scale), round(source_height * scale)
        resized = np.asarray(Image.fromarray(image).resize((resized_width, resized_height), Image.Resampling.BILINEAR))
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = np.asarray(self._session.run(None, {self._input_name: tensor})[0]).squeeze(0)
        if output.shape[0] < output.shape[1]:
            output = output.T
        class_scores = output[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(output)), class_ids]
        keep = scores >= confidence
        output, class_ids, scores = output[keep], class_ids[keep], scores[keep]
        if len(output) == 0:
            return []
        boxes = np.empty((len(output), 4), dtype=np.float32)
        boxes[:, 0] = (output[:, 0] - output[:, 2] / 2 - pad_x) / scale
        boxes[:, 1] = (output[:, 1] - output[:, 3] / 2 - pad_y) / scale
        boxes[:, 2] = (output[:, 0] + output[:, 2] / 2 - pad_x) / scale
        boxes[:, 3] = (output[:, 1] + output[:, 3] / 2 - pad_y) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, source_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, source_height)
        selected: list[int] = []
        for class_id in np.unique(class_ids):
            indices = np.where(class_ids == class_id)[0]
            order = indices[np.argsort(scores[indices])[::-1]]
            while len(order):
                current = int(order[0])
                selected.append(current)
                if len(order) == 1:
                    break
                rest = order[1:]
                left_top = np.maximum(boxes[current, :2], boxes[rest, :2])
                right_bottom = np.minimum(boxes[current, 2:], boxes[rest, 2:])
                intersection = np.prod(np.maximum(0, right_bottom - left_top), axis=1)
                current_area = np.prod(np.maximum(0, boxes[current, 2:] - boxes[current, :2]))
                rest_area = np.prod(np.maximum(0, boxes[rest, 2:] - boxes[rest, :2]), axis=1)
                overlap = intersection / np.maximum(current_area + rest_area - intersection, 1e-12)
                order = rest[overlap <= iou]
        return [
            RawDetection(
                bbox_xyxy=tuple(float(value) for value in boxes[index]),  # type: ignore[arg-type]
                class_id=int(class_ids[index]),
                confidence=float(scores[index]),
            )
            for index in sorted(selected, key=lambda value: scores[value], reverse=True)
        ]


class OnnxEmbeddingModel:
    """ONNX feature extractor with ImageNet-compatible preprocessing."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        if not crops:
            return np.empty((0, 0), dtype=np.float32)
        arrays: list[np.ndarray] = []
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        for crop in crops:
            resized = crop.resize((224, 224), Image.Resampling.BILINEAR)
            value = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            arrays.append((value - mean) / std)
        outputs = self._session.run(None, {self._input_name: np.stack(arrays)})[0]
        return np.asarray(outputs).reshape(len(crops), -1)


@dataclass(frozen=True)
class EngineMetadata:
    detector_version: str
    anomaly_version: str
    runtime: str
    thresholds: dict[str, float]
    checksums: dict[str, str]
    training_data_summary: dict[str, object]


class AnalysisEngine:
    """Compose detection, crop extraction, and calibrated anomaly scoring."""

    def __init__(
        self,
        detector: Detector,
        embedder: Embedder,
        anomaly_index: ConditionalKNNIndex,
        metadata: EngineMetadata,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.anomaly_index = anomaly_index
        self.metadata = metadata

    def analyze(
        self, image: Image.Image, array: np.ndarray, confidence: float, iou: float
    ) -> tuple[list[DetectionPayload], dict[str, float], list[str]]:
        detection_start = time.perf_counter()
        raw = self.detector.predict(array, confidence, iou)
        detection_ms = (time.perf_counter() - detection_start) * 1000
        crops: list[Image.Image] = []
        valid: list[RawDetection] = []
        for detection in raw:
            x1, y1, x2, y2 = detection.bbox_xyxy
            bounded = (
                max(0, min(image.width, round(x1))),
                max(0, min(image.height, round(y1))),
                max(0, min(image.width, round(x2))),
                max(0, min(image.height, round(y2))),
            )
            if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                continue
            crops.append(image.crop(bounded))
            valid.append(detection)
        anomaly_start = time.perf_counter()
        embeddings = self.embedder.embed(crops) if crops else np.empty((0, 0))
        results: list[DetectionPayload] = []
        warnings: list[str] = []
        for position, detection in enumerate(valid):
            denomination = CLASS_TO_DENOMINATION.get(detection.class_id)
            if denomination is None:
                warnings.append(f"Ignored detector class {detection.class_id} outside canonical schema.")
                continue
            try:
                score, threshold, flagged = self.anomaly_index.score(embeddings[position], denomination)
                label = "visual-anomaly-review" if flagged else "within-reference-distribution"
            except InsufficientReferenceError:
                score, threshold, flagged, label = 0.0, self.anomaly_index.quantile, None, "unknown-unreliable"
                warnings.append(f"PKR {denomination} lacks a calibrated anomaly reference.")
            results.append(
                {
                    "id": f"note-{len(results) + 1}",
                    "bbox_xyxy": detection.bbox_xyxy,
                    "denomination_pkr": denomination,
                    "confidence": detection.confidence,
                    "anomaly": {
                        "score": score,
                        "threshold": threshold,
                        "flagged": flagged,
                        "label": label,
                    },
                }
            )
        anomaly_ms = (time.perf_counter() - anomaly_start) * 1000
        return results, {"detection": detection_ms, "anomaly": anomaly_ms}, warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_engine(manifest_path: Path, detector_path: Path, anomaly_path: Path) -> AnalysisEngine:
    """Load only artifacts matching the release manifest checksums."""
    if not manifest_path.is_file():
        raise ModelUnavailableError(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ModelUnavailableError("Model manifest is not marked complete.")
    embedder_path = manifest_path.parent / str(manifest["artifacts"]["embedder"]["filename"])
    paths = {"detector": detector_path, "anomaly_index": anomaly_path, "embedder": embedder_path}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ModelUnavailableError("Missing model artifacts: " + ", ".join(missing))
    for name, path in paths.items():
        expected = manifest["artifacts"][name]["sha256"]
        if _sha256(path) != expected:
            raise ModelUnavailableError(f"Checksum mismatch for {name}: {path}")
    metadata = EngineMetadata(
        detector_version=manifest["detector_version"],
        anomaly_version=manifest["anomaly_version"],
        runtime="onnx-cpu",
        thresholds=manifest["thresholds"],
        checksums={name: manifest["artifacts"][name]["sha256"] for name in paths},
        training_data_summary=manifest.get("training_data_summary", {}),
    )
    return AnalysisEngine(
        detector=OnnxDetector(detector_path),
        embedder=OnnxEmbeddingModel(embedder_path),
        anomaly_index=ConditionalKNNIndex.load(anomaly_path),
        metadata=metadata,
    )


@dataclass(frozen=True)
class ClassifierMetadata:
    """Verified serving metadata for the classification-only product."""

    classifier_version: str
    runtime: str
    temperature: float
    checksums: dict[str, str]
    preprocessing: dict[str, object]
    training_data_summary: dict[str, object]
    evaluation_summary: dict[str, object]
    quality_gate: dict[str, object]


class OnnxClassifier:
    """ONNX Runtime EfficientNet classifier with calibrated probabilities."""

    def __init__(self, model_path: Path, temperature: float) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self.temperature = temperature

    def predict_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Return seven probabilities in canonical denomination order."""
        logits = np.asarray(self._session.run(None, {self._input_name: tensor})[0])[0]
        shifted = logits / self.temperature
        shifted -= shifted.max()
        values = np.exp(shifted)
        return np.asarray(values / values.sum(), dtype=np.float64)


class Classifier(Protocol):
    def predict_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Return canonical denomination probabilities."""


class ClassificationEngine:
    """Classify one banknote image using a verified calibrated model."""

    def __init__(self, classifier: Classifier, metadata: ClassifierMetadata) -> None:
        self.classifier = classifier
        self.metadata = metadata

    def classify(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float]]:
        """Return probabilities with separately measured preprocessing and inference latency."""
        preprocess_started = time.perf_counter()
        tensor = classifier_tensor(image)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        inference_started = time.perf_counter()
        probabilities = np.asarray(self.classifier.predict_tensor(tensor), dtype=np.float64)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        if probabilities.shape != (len(DENOMINATIONS),) or not np.isfinite(probabilities).all():
            raise ValueError("Classifier returned an invalid probability vector.")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            raise ValueError("Classifier probabilities do not sum to one.")
        return probabilities, {"preprocess": preprocess_ms, "inference": inference_ms}


def load_classifier_engine(manifest_path: Path, classifier_path: Path) -> ClassificationEngine:
    """Load a checksum-verified classifier only when its quality gate passed."""
    if not manifest_path.is_file():
        raise ModelUnavailableError(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ModelUnavailableError("Model manifest is not marked complete.")
    if not manifest.get("quality_gate", {}).get("passed", False):
        raise ModelUnavailableError("Classifier exists but did not pass the registered quality gate.")
    expected_order = list(DENOMINATIONS)
    if manifest.get("class_order") != expected_order:
        raise ModelUnavailableError("Release class order does not match the canonical denomination schema.")
    if not classifier_path.is_file():
        raise ModelUnavailableError(f"Missing classifier artifact: {classifier_path}")
    expected = manifest["artifacts"]["classifier"]["sha256"]
    if _sha256(classifier_path) != expected:
        raise ModelUnavailableError(f"Checksum mismatch for classifier: {classifier_path}")
    metadata = ClassifierMetadata(
        classifier_version=str(manifest["classifier_version"]),
        runtime="onnx-cpu",
        temperature=float(manifest["calibration"]["temperature"]),
        checksums={"classifier": expected},
        preprocessing=dict(manifest["preprocessing"]),
        training_data_summary=dict(manifest.get("training_data_summary", {})),
        evaluation_summary=dict(manifest.get("evaluation_summary", {})),
        quality_gate=dict(manifest["quality_gate"]),
    )
    return ClassificationEngine(OnnxClassifier(classifier_path, metadata.temperature), metadata)
