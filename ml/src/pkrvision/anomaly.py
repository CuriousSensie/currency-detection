"""Denomination-conditional k-nearest-neighbour anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class InsufficientReferenceError(RuntimeError):
    """Raised when a denomination lacks enough reference embeddings."""


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.asarray(values / np.clip(norms, 1e-12, None))


@dataclass
class ConditionalKNNIndex:
    """Calibrated anomaly index using cosine distance to class references."""

    references: dict[int, np.ndarray]
    calibration_scores: dict[int, np.ndarray]
    k: int = 5
    quantile: float = 0.95

    @classmethod
    def fit(
        cls,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        validation_embeddings: np.ndarray,
        validation_labels: np.ndarray,
        *,
        k: int = 5,
        quantile: float = 0.95,
    ) -> ConditionalKNNIndex:
        """Fit references and validation-only empirical calibration distributions."""
        index = cls(references={}, calibration_scores={}, k=k, quantile=quantile)
        for denomination in sorted({int(value) for value in train_labels}):
            refs = _normalize(train_embeddings[train_labels == denomination].astype(np.float32))
            validation = validation_embeddings[validation_labels == denomination]
            if len(refs) < k or len(validation) == 0:
                continue
            index.references[denomination] = refs
            index.calibration_scores[denomination] = np.sort(index.raw_scores(validation, denomination))
        return index

    def raw_scores(self, embeddings: np.ndarray, denomination: int) -> np.ndarray:
        """Return mean cosine distance to the k nearest denomination references."""
        if denomination not in self.references:
            raise InsufficientReferenceError(f"No reference index for PKR {denomination}.")
        queries = _normalize(np.atleast_2d(embeddings).astype(np.float32))
        distances = 1.0 - queries @ self.references[denomination].T
        neighbours = np.partition(distances, self.k - 1, axis=1)[:, : self.k]
        return np.asarray(neighbours.mean(axis=1))

    def score(self, embedding: np.ndarray, denomination: int) -> tuple[float, float, bool]:
        """Return empirical percentile, calibrated threshold, and review decision."""
        calibration = self.calibration_scores.get(denomination)
        if calibration is None or len(calibration) == 0:
            raise InsufficientReferenceError(f"No calibration for PKR {denomination}.")
        raw = float(self.raw_scores(np.atleast_2d(embedding), denomination)[0])
        percentile = float(np.searchsorted(calibration, raw, side="right") / len(calibration))
        return percentile, self.quantile, percentile >= self.quantile

    def save(self, path: Path) -> None:
        """Persist the index without executable serialization formats."""
        payload: dict[str, np.ndarray] = {
            "k": np.asarray([self.k]),
            "quantile": np.asarray([self.quantile], dtype=np.float32),
        }
        for denomination, references in self.references.items():
            payload[f"references_{denomination}"] = references
            payload[f"calibration_{denomination}"] = self.calibration_scores[denomination]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: Path) -> ConditionalKNNIndex:
        """Load a previously persisted index."""
        archive = np.load(path, allow_pickle=False)
        references: dict[int, np.ndarray] = {}
        calibration: dict[int, np.ndarray] = {}
        for key in archive.files:
            if key.startswith("references_"):
                denomination = int(key.removeprefix("references_"))
                references[denomination] = archive[key]
                calibration[denomination] = archive[f"calibration_{denomination}"]
        return cls(
            references=references,
            calibration_scores=calibration,
            k=int(archive["k"][0]),
            quantile=float(archive["quantile"][0]),
        )
