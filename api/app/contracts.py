"""Versioned HTTP contracts for single-banknote classification."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelIdentity(StrictModel):
    classifier_version: str
    runtime: str


class ImageInfo(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ClassProbability(StrictModel):
    denomination_pkr: int
    probability: float = Field(ge=0, le=1)


class ClassificationPrediction(StrictModel):
    denomination_pkr: int
    confidence: float = Field(ge=0, le=1)
    probabilities: list[ClassProbability] = Field(min_length=7, max_length=7)


class Timings(StrictModel):
    decode: float = Field(ge=0)
    preprocess: float = Field(ge=0)
    inference: float = Field(ge=0)
    total: float = Field(ge=0)


class ClassificationResponse(StrictModel):
    request_id: str
    model: ModelIdentity
    image: ImageInfo
    prediction: ClassificationPrediction
    timings_ms: Timings
    warnings: list[str]


class ErrorBody(StrictModel):
    code: str
    message: str
    request_id: str
    details: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(StrictModel):
    error: ErrorBody


class HealthResponse(StrictModel):
    status: Literal["alive", "ready", "not_ready"]


class ModelInfoResponse(StrictModel):
    ready: bool
    model: ModelIdentity
    denominations: list[int]
    preprocessing: dict[str, object]
    artifact_checksums: dict[str, str]
    quality_gate: dict[str, object]
    training_data_summary: dict[str, object]
    evaluation_summary: dict[str, object]
    limitations: list[str]
    missing_artifacts: list[str]
