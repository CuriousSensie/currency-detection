"""FastAPI entrypoint for the classification-first PKR Vision product."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from anyio import to_thread
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response

from api.app.contracts import (
    ClassificationPrediction,
    ClassificationResponse,
    ClassProbability,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ImageInfo,
    ModelIdentity,
    ModelInfoResponse,
    Timings,
)
from api.app.settings import Settings
from pkrvision.constants import DENOMINATIONS
from pkrvision.image_io import ImageDecodeError, decode_image
from pkrvision.inference import ClassificationEngine, load_classifier_engine

logger = structlog.get_logger()
LIMITATIONS = [
    "The input must contain one Pakistani banknote denomination or one banknote crop.",
    "Multiple-note detection, counting, totals, and counterfeit verification are outside v1.",
    "Abduls is a public cross-dataset evaluation set, not user-collected real-world validation.",
]


def _error(status: int, code: str, message: str, request_id: str, **details: object) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=request_id, details=details))
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"))


def create_app(settings: Settings | None = None, engine: ClassificationEngine | None = None) -> FastAPI:
    """Create the API with injectable inference for model-free contract tests."""
    config = settings or Settings()
    supplied_engine = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = supplied_engine
        app.state.load_error = None
        app.state.inference_semaphore = asyncio.Semaphore(config.max_concurrent_inferences)
        if supplied_engine is None:
            try:
                app.state.engine = load_classifier_engine(config.model_manifest, config.classifier_path)
            except Exception as exc:
                app.state.load_error = str(exc)
                logger.warning("model_not_ready", reason=str(exc), error_type=type(exc).__name__)
        yield

    app = FastAPI(
        title="PKR Vision Classification API",
        version="1.0.0",
        description="Research API for classifying one Pakistani banknote denomination per image.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(
            422,
            "validation_error",
            "The request parameters are invalid.",
            request.state.request_id,
            errors=exc.errors(),
        )

    @app.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(status="alive")

    @app.get("/health/ready", response_model=HealthResponse)
    async def ready(request: Request) -> HealthResponse | JSONResponse:
        if request.app.state.engine is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return HealthResponse(status="ready")

    @app.get("/v1/model-info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        current: ClassificationEngine | None = request.app.state.engine
        if current is None:
            return ModelInfoResponse(
                ready=False,
                model=ModelIdentity(classifier_version="not-installed", runtime="onnx-cpu"),
                denominations=list(DENOMINATIONS),
                preprocessing={"resize": "aspect-ratio-preserving-letterbox", "size": [224, 224]},
                artifact_checksums={},
                quality_gate={"passed": False},
                training_data_summary={"status": "not_run"},
                evaluation_summary={"status": "not_run"},
                limitations=LIMITATIONS,
                missing_artifacts=[request.app.state.load_error or "Classifier artifacts are unavailable."],
            )
        metadata = current.metadata
        return ModelInfoResponse(
            ready=True,
            model=ModelIdentity(classifier_version=metadata.classifier_version, runtime=metadata.runtime),
            denominations=list(DENOMINATIONS),
            preprocessing=metadata.preprocessing,
            artifact_checksums=metadata.checksums,
            quality_gate=metadata.quality_gate,
            training_data_summary=metadata.training_data_summary,
            evaluation_summary=metadata.evaluation_summary,
            limitations=LIMITATIONS,
            missing_artifacts=[],
        )

    @app.post(
        "/v1/classify",
        response_model=ClassificationResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def classify(
        request: Request,
        image: UploadFile = File(description="JPEG, PNG, or WebP image containing one banknote denomination"),
    ) -> ClassificationResponse | JSONResponse:
        request_id = request.state.request_id
        current: ClassificationEngine | None = request.app.state.engine
        if current is None:
            return _error(
                503,
                "model_not_ready",
                "A verified classifier that passed the quality gate is not installed.",
                request_id,
                recovery="Run the documented training, evaluation, export, and release workflow.",
            )
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            return _error(
                415,
                "unsupported_media_type",
                "Use a JPEG, PNG, or WebP upload.",
                request_id,
                received=image.content_type,
            )
        payload = await image.read(config.max_upload_bytes + 1)
        if len(payload) > config.max_upload_bytes:
            return _error(413, "upload_too_large", "The image exceeds the configured byte limit.", request_id)
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        try:
            decoded = decode_image(payload, config.max_image_pixels)
        except ImageDecodeError as exc:
            return _error(400, "invalid_image", str(exc), request_id)
        decode_ms = (time.perf_counter() - decode_start) * 1000
        try:
            async with request.app.state.inference_semaphore:
                probabilities, model_timings = await to_thread.run_sync(current.classify, decoded.image)
        except Exception as exc:
            logger.exception("classification_failed", request_id=request_id, error_type=type(exc).__name__)
            return _error(
                500,
                "classification_failed",
                "The classifier could not process this image.",
                request_id,
            )
        class_id = int(probabilities.argmax())
        ordered = [
            ClassProbability(denomination_pkr=denomination, probability=float(probabilities[index]))
            for index, denomination in enumerate(DENOMINATIONS)
        ]
        return ClassificationResponse(
            request_id=request_id,
            model=ModelIdentity(
                classifier_version=current.metadata.classifier_version,
                runtime=current.metadata.runtime,
            ),
            image=ImageInfo(width=decoded.image.width, height=decoded.image.height),
            prediction=ClassificationPrediction(
                denomination_pkr=DENOMINATIONS[class_id],
                confidence=float(probabilities[class_id]),
                probabilities=ordered,
            ),
            timings_ms=Timings(
                decode=decode_ms,
                preprocess=model_timings["preprocess"],
                inference=model_timings["inference"],
                total=(time.perf_counter() - total_start) * 1000,
            ),
            warnings=[],
        )

    @app.post("/v1/analyze", status_code=410, response_model=ErrorResponse)
    async def deprecated_analyze(request: Request) -> JSONResponse:
        return _error(
            410,
            "classification_only_api",
            "Multi-note analysis is not part of v1. Submit one banknote to POST /v1/classify.",
            request.state.request_id,
        )

    return app


app = create_app()
