# Architecture

The classification data builder converts unanimous UCP labels into deterministic image-level splits and keeps Abduls crops in an isolated external partition. Training consumes only manifest-selected IDs, fits calibration on validation logits, and writes prediction-level evidence.

The selected EfficientNet checkpoint exports to ONNX. The release manifest checks the ONNX file, model card, result summary, preprocessing, calibration temperature, class order, dataset hash, and quality gate. ONNX Runtime is the CPU serving path; PyTorch remains the research reference.

FastAPI owns input safety and the versioned contract. The Next.js console consumes only HTTP responses and checked-in result records; it cannot calculate or fabricate model metrics. Detector and anomaly modules are inactive future work and are not exposed by the v1 API, release bundle, or console.
