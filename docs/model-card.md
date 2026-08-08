# EfficientNet-B0 denomination classifier card

## Status

Trained classifier available from the Kaggle run archived locally on 2026-09-11. The selected checkpoint passed the preregistered internal, external, and per-class recall quality gate. Local CPU benchmark evidence has been recorded.

## Intended use

Classify one visible Pakistani banknote denomination or one note crop into PKR 10, 20, 50, 100, 500, 1000, or 5000. Return all calibrated probabilities so competing classes remain inspectable.

The measured local CPU warm combined latency is below 20 ms, so the classifier is suitable for live-feed classification experiments when each processed frame already contains one visible note or a single-note crop.

## Out of scope

Multiple-note localization and counting, monetary totals, condition grading, visual-anomaly scoring, front/back recognition, PKR 75, and authenticity verification.

## Model and preprocessing

The candidate is ImageNet-pretrained EfficientNet-B0. EXIF orientation is normalized, the RGB image is fitted into a 224×224 canvas without aspect-ratio distortion, and ImageNet normalization is applied. Validation-only temperature scaling calibrates logits.

## Evidence and release

Primary evaluation uses the untouched UCP test split. Abduls crops provide cross-dataset evaluation, not real-world validation. The serving manifest records class order, preprocessing, calibration, data and artifact hashes, internal/external summaries, and the preregistered quality-gate outcome. A failed gate leaves the API not ready.

## Selected checkpoint

- Condition: full training set with registered augmentation.
- Selection rule: higher validation macro F1.
- Checkpoint SHA-256: `c2f8fd459dea6585f84d62a0d29bf9f3f17b9ffd44633ba07f2294392ad6204c`.
- ONNX SHA-256: `804460c6a53d8606211fb42420eb9160ceb9c5ba41f4ee8e8da131188ea1a56e`.
- Calibration temperature: `0.05`.

## Verified metrics

- UCP test accuracy: `0.9963`.
- UCP test macro F1: `0.9962`.
- UCP test balanced accuracy: `0.9962`.
- UCP test expected calibration error: `0.0044`.
- UCP test top-2 accuracy: `1.0000`.
- Abduls cross-dataset accuracy: `0.9755`.
- Abduls cross-dataset macro F1: `0.9721`.
- Abduls cross-dataset expected calibration error: `0.0245`.
- ONNX/PyTorch top-1 agreement on UCP test: `1.0000`.
- Local CPU warm combined latency mean: `6.91 ms`.
- Local CPU warm combined latency p95: `8.53 ms`.
- Local CPU sequential throughput: `144.7 images/s`.

The UCP test macro F1 95% bootstrap interval is `[0.9905, 1.0000]`. The Abduls macro F1 95% bootstrap interval is `[0.9521, 0.9894]`. These intervals describe fixed-test resampling only and exclude training-seed variance.
