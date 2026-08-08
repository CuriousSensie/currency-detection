# Release reproduction

1. Verify source acquisition records and regenerate the classification manifest.
2. Confirm zero duplicate-cluster leakage and review the UCP/Abduls audit summaries.
3. Execute every planned ablation run from the pinned Colab commit.
4. Select the full-data condition using validation evidence only and evaluate that checkpoint on Abduls.
5. Export `classifier.onnx`; create `onnx-validation.json` with `validate-classifier-onnx` over the complete fixed test set.
6. Confirm the equivalence report passed and binds the selected checkpoint hash, ONNX hash, canonical class order, complete top-one agreement, and bounded calibrated-probability drift.
7. Add a completed model card and run result summary beside the ONNX file.
8. Run `pkrvision build-release`; the manifest records checksums and computes the fixed quality gate.
9. Install the bundle under `artifacts/models/current`, run API/console end-to-end checks, and use `benchmark-classifier` to measure cold start, warm latency, sequential throughput, peak memory, and size on named hardware and sample files.

Never publish a placeholder under a production artifact name. A complete bundle that misses the quality gate remains research evidence but cannot make `/health/ready` succeed.
