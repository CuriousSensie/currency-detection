# API

`POST /v1/classify` accepts one multipart field named `image`. JPEG, PNG, and WebP are supported up to 12 MB encoded and 20 megapixels decoded. The image must contain one denomination.

The response contains request and model identity, source dimensions, predicted denomination, calibrated confidence, seven probabilities in canonical order, separate decode/preprocess/inference/total timings, and warnings. Images are not persisted.

`GET /v1/model-info` exposes readiness, classifier version, class order, preprocessing, checksums, quality-gate evidence, data/evaluation summaries, limitations, and missing artifacts. Health endpoints separate process liveness from model readiness.

`POST /v1/analyze` returns HTTP 410 with code `classification_only_api`. Every error uses `{error: {code, message, request_id, details}}`.
