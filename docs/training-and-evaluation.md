# Training and evaluation

Use a pinned notebook and keep the canonical classification directory immutable. Kaggle is the preferred free-GPU path when Colab session or Drive limits interrupt the run. Record the Git commit, configuration and dataset-manifest hashes, Python/package versions, GPU details, duration, checkpoint hash, and failure state.

## Kaggle workflow

1. Create a private Kaggle Dataset containing `classification-v1.tar.gz` or the extracted `classification-v1/` directory. Kaggle may expose the extracted directory at a path such as `/kaggle/input/datasets/hasnaatk/pkr-classification-v1/classification-v1`.
2. Create a Kaggle Notebook, enable internet, attach that Dataset, and enable GPU.
3. Upload or paste `notebooks/pkr_vision_kaggle.ipynb`.
4. Set `REPOSITORY_URL`, `COMMIT`, and `KAGGLE_DATASET_SLUG` in the first code cell.
5. Run cells in order. Outputs are written under `/kaggle/working/artifacts`.
6. Download `/kaggle/working/pkr-classification-results.tar.gz` or save a Kaggle Notebook version so the output artifact is retained.

The Kaggle notebook verifies the dataset archive checksum and manifest hash before training. Its training cell skips any run directory that already contains `run.json`, which lets a restarted session continue from previously restored outputs.

## Generic commands

Generate the immutable experiment plan:

```bash
pkrvision plan-classification-ablation data/processed/classification-v1/samples.jsonl artifacts/mlruns/classification/plan.json
```

Run every planned level under both conditions. For example:

```bash
pkrvision train-classifier configs/research/ablation.yaml data/processed/classification-v1/internal artifacts/mlruns/classification/plan.json 100 artifacts/mlruns/classification/n100-noaug
pkrvision train-classifier configs/research/ablation.yaml data/processed/classification-v1/internal artifacts/mlruns/classification/plan.json 100 artifacts/mlruns/classification/n100-aug --augmentation
```

At any point, generate the machine-readable run ledger:

```bash
pkrvision audit-classification-runs artifacts/mlruns/classification/plan.json artifacts/mlruns/classification artifacts/mlruns/classification/run-matrix.json
```

This ledger is derived from the registered plan and run directories. It marks each condition as `complete`, `failed`, `not_run`, or `omitted`, and rejects a completed run whose level, augmentation flag, or subset hash does not match the plan.

After the two full-data runs, use `select-classifier`, evaluate only the selected checkpoint with `evaluate-external`, then export and validate it:

```bash
pkrvision export-classifier artifacts/mlruns/classification/full-selected/best.pt artifacts/models/release/classifier.onnx
pkrvision validate-classifier-onnx artifacts/mlruns/classification/full-selected/best.pt artifacts/models/release/classifier.onnx data/processed/classification-v1/internal/test artifacts/models/release/onnx-validation.json
```

Validation compares every fixed-test prediction against PyTorch, requires complete top-1 agreement, checks the maximum calibrated-probability drift, exercises dynamic batches, and binds the report to both artifact hashes. Prediction NPZ files are canonical evidence; result prose is generated only after result validation.

Grad-CAM and error-review outputs must reference sample IDs and checkpoint hashes. They are qualitative diagnostics, not additional test-set selection signals.

```bash
pkrvision generate-gradcam artifacts/mlruns/classification/full-selected/best.pt data/processed/classification-v1/internal/test artifacts/mlruns/classification/full-selected/test-predictions.npz artifacts/mlruns/classification/full-selected/gradcam
```

The command selects the most confident fixed-test errors (or the least confident correct cases when there are no errors), writes overlays and checksums, summarizes denomination confusions, and labels the output as post-evaluation diagnosis only.

Benchmark the release on the named deployment machine with representative fixed-test files:

```bash
pkrvision benchmark-classifier artifacts/models/release/classifier.onnx 1.0 --image path/to/test-1.jpg --image path/to/test-2.jpg artifacts/results/cpu-benchmark-complete.json
```

Replace `1.0` with the selected run's recorded validation temperature. The report binds sample and model checksums and separates session load, first inference, warm preprocessing, warm inference, sequential throughput, peak RSS, and model size. It does not include image decoding or claim concurrent-server throughput.
