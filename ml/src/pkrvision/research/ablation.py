"""Nested sample selection for the registered data-efficiency study."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pkrvision.constants import PROJECT_SEED


@dataclass(frozen=True)
class AblationLevel:
    requested_per_class: int | str
    status: str
    reason: str | None
    sample_ids: tuple[str, ...]


def build_nested_levels(
    sample_ids: list[str], labels: list[int], levels: tuple[int, ...] = (10, 50, 100, 500), seed: int = PROJECT_SEED
) -> list[AblationLevel]:
    """Create deterministic, class-balanced subsets where larger levels contain smaller ones."""
    grouped: dict[int, list[str]] = defaultdict(list)
    for sample_id, label in zip(sample_ids, labels, strict=True):
        grouped[label].append(sample_id)
    rng = np.random.default_rng(seed)
    for label in grouped:
        grouped[label] = list(np.asarray(grouped[label])[rng.permutation(len(grouped[label]))])
    minimum = min((len(values) for values in grouped.values()), default=0)
    output: list[AblationLevel] = []
    for level in levels:
        if level > minimum:
            output.append(AblationLevel(level, "omitted", f"smallest class has {minimum} training samples", ()))
            continue
        selected = tuple(sorted(sample for values in grouped.values() for sample in values[:level]))
        output.append(AblationLevel(level, "planned", None, selected))
    return output


def append_full_level(levels: list[AblationLevel], sample_ids: list[str]) -> list[AblationLevel]:
    """Append the registered all-training-samples condition."""
    return [*levels, AblationLevel("full", "planned", None, tuple(sorted(sample_ids)))]


def subset_digest(level: AblationLevel) -> str:
    """Return a stable digest recorded with each experiment."""
    return hashlib.sha256("\n".join(level.sample_ids).encode()).hexdigest()


def audit_classification_run_matrix(plan_path: Path, runs_root: Path, output: Path) -> Path:
    """Account for every registered classification ablation condition."""
    plan: list[dict[str, Any]] = json.loads(plan_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    counts = {"complete": 0, "failed": 0, "not_run": 0, "omitted": 0}
    for level in plan:
        requested = level["requested_per_class"]
        for augmentation in (False, True):
            suffix = "aug" if augmentation else "noaug"
            run_dir = runs_root / f"n{requested}-{suffix}"
            row: dict[str, Any] = {
                "requested_per_class": requested,
                "augmentation": augmentation,
                "run_dir": str(run_dir),
                "planned_subset_sha256": level.get("subset_sha256"),
            }
            if level.get("status") != "planned":
                row.update({"status": "omitted", "reason": level.get("reason")})
                counts["omitted"] += 1
                rows.append(row)
                continue
            run_json = run_dir / "run.json"
            failure_json = run_dir / "failure.json"
            if run_json.is_file():
                record = json.loads(run_json.read_text(encoding="utf-8"))
                errors = []
                if record.get("status") != "complete":
                    errors.append("run_json_status_not_complete")
                if record.get("samples_per_class") != requested:
                    errors.append("samples_per_class_mismatch")
                if bool(record.get("augmentation")) != augmentation:
                    errors.append("augmentation_mismatch")
                if record.get("subset_sha256") != level.get("subset_sha256"):
                    errors.append("subset_sha256_mismatch")
                status = "failed" if errors else "complete"
                row.update(
                    {
                        "status": status,
                        "errors": errors,
                        "checkpoint_sha256": record.get("checkpoint_sha256"),
                        "config_sha256": record.get("config_sha256"),
                        "validation_macro_f1": record.get("validation_metrics", {}).get("macro_f1"),
                        "test_macro_f1": record.get("test_metrics", {}).get("macro_f1"),
                    }
                )
                counts[status] += 1
            elif failure_json.is_file():
                failure = json.loads(failure_json.read_text(encoding="utf-8"))
                row.update(
                    {
                        "status": "failed",
                        "error_type": failure.get("error_type"),
                        "message": failure.get("message"),
                    }
                )
                counts["failed"] += 1
            else:
                row.update({"status": "not_run"})
                counts["not_run"] += 1
            rows.append(row)
    payload = {
        "status": "complete",
        "kind": "classification_ablation_run_matrix",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "runs_root": str(runs_root),
        "counts": counts,
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output
