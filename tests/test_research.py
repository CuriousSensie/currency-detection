import json

import numpy as np
import pytest

from pkrvision.research.ablation import audit_classification_run_matrix, build_nested_levels, subset_digest
from pkrvision.research.classifier import calibrated_probabilities, fit_temperature
from pkrvision.research.explainability import _selected_indices
from pkrvision.research.metrics import bootstrap_accuracy_interval, bootstrap_intervals, classification_metrics


def test_ablation_levels_are_nested_and_infeasible_levels_are_explicit() -> None:
    ids = [f"{label}-{index}" for label in (0, 1) for index in range(60)]
    labels = [label for label in (0, 1) for _ in range(60)]
    levels = build_nested_levels(ids, labels, levels=(10, 50, 100), seed=7)
    assert [level.status for level in levels] == ["planned", "planned", "omitted"]
    assert set(levels[0].sample_ids) < set(levels[1].sample_ids)
    assert levels[2].reason == "smallest class has 60 training samples"


def test_metrics_match_small_fixture() -> None:
    targets = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([0, 1, 1, 1])
    probabilities = np.asarray([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    metrics = classification_metrics(targets, probabilities, [0, 1])
    assert metrics.accuracy == 0.75
    assert metrics.balanced_accuracy == 0.75
    low, high = bootstrap_accuracy_interval(targets, predictions, iterations=100, seed=3)
    assert 0 <= low <= high <= 1
    intervals = bootstrap_intervals(targets, probabilities, iterations=20, seed=3)
    assert set(intervals) == {"accuracy", "balanced_accuracy", "macro_f1"}


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.asarray([]), np.empty((0, 2)), [0, 1])


def test_temperature_scaling_returns_probabilities() -> None:
    logits = np.asarray([[4.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    targets = np.asarray([0, 1, 1])
    temperature = fit_temperature(logits, targets)
    probabilities = calibrated_probabilities(logits, temperature)
    assert temperature > 0
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_gradcam_review_prioritizes_confident_errors() -> None:
    targets = np.asarray([0, 0, 1])
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    assert _selected_indices(targets, probabilities, 2) == [1]


def test_classification_run_matrix_accounts_for_registered_conditions(tmp_path) -> None:
    planned = build_nested_levels(["a", "b"], [0, 1], levels=(1, 2), seed=3)
    plan = [
        {
            "requested_per_class": level.requested_per_class,
            "status": level.status,
            "reason": level.reason,
            "sample_ids": list(level.sample_ids),
            "subset_sha256": subset_digest(level),
        }
        for level in planned
    ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    run_dir = tmp_path / "runs" / "n1-noaug"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "samples_per_class": 1,
                "augmentation": False,
                "subset_sha256": plan[0]["subset_sha256"],
                "checkpoint_sha256": "abc",
                "config_sha256": "def",
                "validation_metrics": {"macro_f1": 0.7},
                "test_metrics": {"macro_f1": 0.6},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "matrix.json"
    audit_classification_run_matrix(plan_path, tmp_path / "runs", output)
    matrix = json.loads(output.read_text(encoding="utf-8"))
    assert matrix["counts"] == {"complete": 1, "failed": 0, "not_run": 1, "omitted": 2}
    assert matrix["runs"][0]["status"] == "complete"
    assert matrix["runs"][1]["status"] == "not_run"
    assert matrix["runs"][2]["status"] == "omitted"
