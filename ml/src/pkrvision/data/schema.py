"""Versioned dataset provenance contracts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceDefinition(BaseModel):
    """Expected identity and licensing of a source archive."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    url: str
    version: int | str
    retrieved_at: date | None = None
    declared_license: str
    expected_classes: list[str]
    archive_sha256: str | None = None
    local_archive: Path | None = None


class Box(BaseModel):
    """Canonical normalized YOLO box."""

    model_config = ConfigDict(extra="forbid")
    class_id: int = Field(ge=0)
    x_center: float = Field(ge=0, le=1)
    y_center: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @field_validator("width", "height")
    @classmethod
    def nonzero(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Box dimensions must be positive.")
        return value


class SampleRecord(BaseModel):
    """One auditable image from source bytes to derived split."""

    model_config = ConfigDict(extra="forbid")
    sample_id: str
    source_id: str
    source_relative_path: str
    image_sha256: str
    perceptual_hash: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    boxes: list[Box]
    duplicate_cluster_id: str | None = None
    split: Literal["train", "validation", "test"] | None = None
    status: Literal["accepted", "excluded", "invalid"]
    reason: str | None = None


class AuditSummary(BaseModel):
    """Reconciled counts emitted by the audit command."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    total_images: int
    accepted_images: int
    excluded_images: int
    invalid_images: int
    class_counts: dict[str, int]
    reasons: dict[str, int]
