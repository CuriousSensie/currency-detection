from pathlib import Path

import pytest
from pydantic import ValidationError

from pkrvision.results import ExperimentResult, validate_result_directory


def test_not_run_result_cannot_contain_metrics() -> None:
    with pytest.raises(ValidationError):
        ExperimentResult(
            experiment_id="fake",
            status="not_run",
            kind="detection",
            metrics={"map50": 0.99},
        )


def test_complete_result_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        ExperimentResult(
            experiment_id="unsupported",
            status="complete",
            kind="detection",
            metrics={"map50": 0.9},
        )


def test_checked_in_result_records_are_valid() -> None:
    results = validate_result_directory(Path("artifacts/results"))
    assert len(results) == 3
    for result in results:
        if result.status == "complete":
            assert result.config_sha256
            assert result.dataset_manifest_sha256
            assert result.checkpoint_sha256
            assert result.metrics
