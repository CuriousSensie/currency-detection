from io import BytesIO

import numpy as np
from api.app.main import create_app
from api.app.settings import Settings
from fastapi.testclient import TestClient
from PIL import Image

from pkrvision.inference import ClassificationEngine, ClassifierMetadata


class StubClassifier:
    def predict_tensor(self, tensor: np.ndarray) -> np.ndarray:
        assert tensor.shape == (1, 3, 224, 224)
        return np.asarray([0.01, 0.02, 0.03, 0.04, 0.85, 0.03, 0.02])


def make_engine() -> ClassificationEngine:
    return ClassificationEngine(
        StubClassifier(),
        ClassifierMetadata(
            classifier_version="fixture-classifier",
            runtime="test-double",
            temperature=1.0,
            checksums={"classifier": "a" * 64},
            preprocessing={"size": [224, 224]},
            training_data_summary={},
            evaluation_summary={},
            quality_gate={"passed": True},
        ),
    )


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (12, 10), "white").save(stream, format="PNG")
    return stream.getvalue()


def test_liveness_does_not_imply_model_readiness() -> None:
    app = create_app(Settings(model_manifest="does-not-exist.json"))
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        assert client.get("/health/ready").status_code == 503
        info = client.get("/v1/model-info").json()
        assert info["ready"] is False
        assert info["quality_gate"] == {"passed": False}


def test_classify_contract_with_injected_engine() -> None:
    app = create_app(engine=make_engine())
    with TestClient(app) as client:
        response = client.post("/v1/classify", files={"image": ("fixture.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["denomination_pkr"] == 500
    assert len(body["prediction"]["probabilities"]) == 7
    assert abs(sum(item["probability"] for item in body["prediction"]["probabilities"]) - 1) < 1e-6
    assert body["model"]["runtime"] == "test-double"
    assert set(body["timings_ms"]) == {"decode", "preprocess", "inference", "total"}


def test_internal_classifier_failure_uses_stable_error() -> None:
    class BrokenClassifier:
        def predict_tensor(self, tensor: np.ndarray) -> np.ndarray:
            raise RuntimeError("private failure detail")

    engine = make_engine()
    engine.classifier = BrokenClassifier()
    app = create_app(engine=engine)
    with TestClient(app) as client:
        response = client.post("/v1/classify", files={"image": ("fixture.png", image_bytes(), "image/png")})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "classification_failed"
    assert "private failure detail" not in response.text


def test_analyze_is_explicitly_deprecated() -> None:
    app = create_app(engine=make_engine())
    with TestClient(app) as client:
        response = client.post("/v1/analyze")
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "classification_only_api"


def test_invalid_image_uses_stable_error_envelope() -> None:
    app = create_app(engine=make_engine())
    with TestClient(app) as client:
        response = client.post("/v1/classify", files={"image": ("bad.png", b"invalid", "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"
    assert response.json()["error"]["request_id"]


def test_incorrect_media_type_is_rejected_before_decode() -> None:
    app = create_app(engine=make_engine())
    with TestClient(app) as client:
        response = client.post(
            "/v1/classify",
            files={"image": ("fixture.bin", image_bytes(), "application/octet-stream")},
        )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
