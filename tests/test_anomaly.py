from pathlib import Path

import numpy as np
import pytest

from pkrvision.anomaly import ConditionalKNNIndex, InsufficientReferenceError


def test_conditional_index_is_class_isolated_and_round_trips(tmp_path: Path) -> None:
    train = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0], [0.01, 0.99], [0.02, 0.98]])
    labels = np.asarray([10, 10, 10, 20, 20, 20])
    validation = np.asarray([[0.97, 0.03], [0.03, 0.97]])
    validation_labels = np.asarray([10, 20])
    index = ConditionalKNNIndex.fit(train, labels, validation, validation_labels, k=2)
    score, threshold, flagged = index.score(np.asarray([0.0, 1.0]), 10)
    assert score == 1.0
    assert threshold == pytest.approx(0.95)
    assert flagged is True
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = ConditionalKNNIndex.load(path)
    loaded_score, loaded_threshold, loaded_flagged = loaded.score(np.asarray([0.0, 1.0]), 10)
    assert loaded_score == pytest.approx(score)
    assert loaded_threshold == pytest.approx(threshold)
    assert loaded_flagged is flagged


def test_missing_class_has_no_synthetic_fallback() -> None:
    index = ConditionalKNNIndex({}, {}, k=2)
    with pytest.raises(InsufficientReferenceError):
        index.score(np.asarray([1.0, 0.0]), 5000)
