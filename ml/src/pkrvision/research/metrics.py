"""Dependency-light classification metrics and fixed-test bootstrap intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    expected_calibration_error: float
    brier_score: float
    top2_accuracy: float
    per_class_precision: dict[int, float]
    per_class_recall: dict[int, float]
    per_class_f1: dict[int, float]
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible metric values."""
        return asdict(self)


def classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, classes: list[int], bins: int = 10
) -> ClassificationMetrics:
    """Compute registered multiclass metrics from a probability matrix."""
    if len(targets) == 0 or probabilities.shape != (len(targets), len(classes)):
        raise ValueError("Targets and class probabilities must be non-empty and aligned.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one.")
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[int(target), int(prediction)] += 1
    recalls = np.divide(np.diag(matrix), matrix.sum(axis=1), out=np.zeros(len(classes)), where=matrix.sum(axis=1) > 0)
    precisions = np.divide(
        np.diag(matrix), matrix.sum(axis=0), out=np.zeros(len(classes)), where=matrix.sum(axis=0) > 0
    )
    f1 = np.divide(
        2 * precisions * recalls, precisions + recalls, out=np.zeros(len(classes)), where=(precisions + recalls) > 0
    )
    correctness = predictions == targets
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in pairwise(edges):
        mask = (confidences >= lower) & (confidences < upper if upper < 1 else confidences <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean()))
    one_hot = np.eye(len(classes), dtype=np.float64)[targets.astype(int)]
    top2 = np.argpartition(probabilities, -2, axis=1)[:, -2:]
    return ClassificationMetrics(
        accuracy=float(correctness.mean()),
        balanced_accuracy=float(recalls.mean()),
        macro_f1=float(f1.mean()),
        expected_calibration_error=ece,
        brier_score=float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        top2_accuracy=float(np.mean([target in choices for target, choices in zip(targets, top2, strict=True)])),
        per_class_precision={value: float(precisions[index]) for index, value in enumerate(classes)},
        per_class_recall={value: float(recalls[index]) for index, value in enumerate(classes)},
        per_class_f1={value: float(f1[index]) for index, value in enumerate(classes)},
        confusion_matrix=matrix.tolist(),
    )


def bootstrap_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 20260908,
) -> dict[str, tuple[float, float]]:
    """Stratified percentile CIs over fixed predictions, excluding seed variance."""
    if len(targets) == 0 or len(probabilities) != len(targets):
        raise ValueError("Targets and probabilities must be non-empty and aligned.")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(targets == value) for value in sorted(np.unique(targets))]
    scores = {name: np.empty(iterations) for name in ("accuracy", "balanced_accuracy", "macro_f1")}
    classes = list(range(probabilities.shape[1]))
    for index in range(iterations):
        sample = np.concatenate([rng.choice(values, len(values), replace=True) for values in class_indices])
        metrics = classification_metrics(targets[sample], probabilities[sample], classes)
        scores["accuracy"][index] = metrics.accuracy
        scores["balanced_accuracy"][index] = metrics.balanced_accuracy
        scores["macro_f1"][index] = metrics.macro_f1
    return {
        name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) for name, values in scores.items()
    }


def bootstrap_accuracy_interval(
    targets: np.ndarray, predictions: np.ndarray, *, iterations: int = 2000, seed: int = 20260908
) -> tuple[float, float]:
    """Return an accuracy CI for compatibility with the original public helper."""
    if len(targets) != len(predictions) or len(targets) == 0:
        raise ValueError("Targets and predictions must be non-empty and aligned.")
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, len(targets), len(targets))
        scores[index] = np.mean(targets[sample] == predictions[sample])
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))
