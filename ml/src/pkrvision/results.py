"""Machine-verifiable experiment result records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentResult(BaseModel):
    """Canonical result that cannot claim completion without provenance."""

    model_config = ConfigDict(extra="forbid")
    experiment_id: str
    status: Literal["not_run", "running", "complete", "failed"]
    kind: Literal["detection", "classification_ablation", "anomaly", "benchmark", "external"]
    config_sha256: str | None = None
    dataset_manifest_sha256: str | None = None
    checkpoint_sha256: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def completed_has_evidence(self) -> ExperimentResult:
        if self.status == "complete":
            missing = [
                name
                for name in ("config_sha256", "dataset_manifest_sha256", "checkpoint_sha256")
                if getattr(self, name) is None
            ]
            if missing or not self.metrics:
                raise ValueError(f"Completed results require provenance and metrics; missing {missing}.")
        if self.status == "not_run" and self.metrics:
            raise ValueError("A not-run experiment cannot contain metrics.")
        return self


def validate_result_directory(path: Path) -> list[ExperimentResult]:
    """Validate every JSON result and ensure experiment IDs are unique."""
    results = [ExperimentResult.model_validate_json(item.read_text()) for item in sorted(path.glob("*.json"))]
    ids = [result.experiment_id for result in results]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique.")
    return results


def write_result(path: Path, result: ExperimentResult) -> None:
    """Write a validated canonical result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
