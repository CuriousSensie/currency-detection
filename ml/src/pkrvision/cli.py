"""Command-line interface for reproducible data and model workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, cast

import typer
import yaml

from pkrvision.anomaly_pipeline import (
    evaluate_anomaly_proxy,
    export_embedder,
    extract_embeddings,
    fit_anomaly_index,
)
from pkrvision.data.acquisition import directory_sha256, download_roboflow, extract_archive
from pkrvision.data.audit import audit_source, prepare_dataset
from pkrvision.data.classification import prepare_classification_dataset
from pkrvision.evaluation import benchmark_classifier, benchmark_detector, choose_detector, evaluate_detector
from pkrvision.release import build_release_manifest
from pkrvision.research.ablation import (
    append_full_level,
    audit_classification_run_matrix,
    build_nested_levels,
    subset_digest,
)
from pkrvision.research.classifier import (
    evaluate_external,
    export_classifier,
    select_classifier,
    train_classifier,
    validate_onnx_classifier,
)
from pkrvision.research.explainability import generate_gradcam_report
from pkrvision.results import validate_result_directory
from pkrvision.training import export_detector, train_detector

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _verified_source_metadata(source_id: str, root: Path, source_config: Path) -> dict[str, object]:
    """Load one approved source declaration and verify its acquisition identity."""
    source_document = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    known_sources = {item["source_id"]: item for item in source_document.get("sources", [])}
    if source_id not in known_sources:
        raise typer.BadParameter(f"{source_id} is not declared in {source_config}.")
    item = known_sources[source_id]
    inclusion = str(item.get("inclusion", ""))
    if item.get("declared_license") == "unverified" or not inclusion.startswith("included_"):
        raise typer.BadParameter(f"{source_id} is not approved for use (inclusion={inclusion!r}).")
    acquisition_path = root / "acquisition.json"
    if not acquisition_path.is_file():
        raise typer.BadParameter(f"{source_id} has no acquisition.json provenance record.")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    configured_version = str(item.get("version"))
    acquired_version = acquisition.get("version")
    if acquired_version is not None and str(acquired_version) != configured_version:
        raise typer.BadParameter(
            f"{source_id} version mismatch: configured {configured_version}, acquired {acquired_version}."
        )
    observed_tree_sha256 = directory_sha256(root)
    recorded_tree_sha256 = acquisition.get("extracted_tree_sha256")
    if recorded_tree_sha256 is not None and recorded_tree_sha256 != observed_tree_sha256:
        raise typer.BadParameter(f"{source_id} extracted-tree checksum does not match acquisition.json.")
    return {**dict(item), "acquisition": acquisition, "verified_tree_sha256": observed_tree_sha256}


@app.command("download-roboflow")
def download_roboflow_command(
    workspace: str,
    project: str,
    version: int,
    destination: Path,
) -> None:
    """Download one pinned Roboflow dataset version using ROBOFLOW_API_KEY."""
    record = download_roboflow(workspace, project, version, os.environ.get("ROBOFLOW_API_KEY", ""), destination)
    typer.echo(json.dumps(record, indent=2))


@app.command("extract-archive")
def extract_archive_command(
    archive: Path,
    destination: Path,
    expected_sha256: str | None = None,
) -> None:
    """Verify and safely import a manually downloaded source archive."""
    typer.echo(json.dumps(extract_archive(archive, destination, expected_sha256), indent=2))


@app.command("audit-source")
def audit_source_command(
    root: Path,
    source_id: str,
    source_config: Path = Path("configs/data/sources.yaml"),
) -> None:
    """Audit a declared YOLO source without writing derived data."""
    source_document = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    known_sources = {item["source_id"]: item for item in source_document.get("sources", [])}
    if source_id not in known_sources:
        raise typer.BadParameter(f"{source_id} is not declared in {source_config}.")
    expected_classes = known_sources[source_id].get("expected_classes")
    if not isinstance(expected_classes, list) or not all(isinstance(item, str) for item in expected_classes):
        raise typer.BadParameter(f"{source_id} has no valid expected_classes declaration.")
    _, summary = audit_source(root, source_id, expected_classes)
    typer.echo(summary.model_dump_json(indent=2))


@app.command("prepare-data")
def prepare_data_command(
    source: Annotated[list[str], typer.Option(help="Repeat as SOURCE_ID=/absolute/extracted/path")],
    output: Path = Path("data/processed/canonical-v1"),
    source_config: Path = Path("configs/data/sources.yaml"),
    max_hamming: int = 4,
) -> None:
    """Build the canonical leakage-resistant dataset."""
    sources: dict[str, Path] = {}
    for value in source:
        if "=" not in value:
            raise typer.BadParameter("Each --source must be SOURCE_ID=PATH.")
        source_id, raw_path = value.split("=", 1)
        sources[source_id] = Path(raw_path).resolve()
    metadata: dict[str, dict[str, object]] = {}
    for source_id in sources:
        metadata[source_id] = _verified_source_metadata(source_id, sources[source_id], source_config)
    typer.echo(
        json.dumps(
            prepare_dataset(
                sources,
                output,
                source_metadata=metadata,
                max_hamming=max_hamming,
            ),
            indent=2,
        )
    )


@app.command("prepare-classification-data")
def prepare_classification_data_command(
    ucp_root: Path,
    external_root: Path,
    output: Path = Path("data/processed/classification-v1"),
    source_config: Path = Path("configs/data/sources.yaml"),
    max_hamming: int = 4,
) -> None:
    """Build UCP internal splits plus an isolated Abduls crop evaluation set."""
    ucp_id = "ucp-pakistani-currency-note-data"
    external_id = "abduls-pkr-notes"
    ucp_metadata = _verified_source_metadata(ucp_id, ucp_root.resolve(), source_config)
    external_metadata = _verified_source_metadata(external_id, external_root.resolve(), source_config)
    typer.echo(
        json.dumps(
            prepare_classification_dataset(
                ucp_root.resolve(),
                external_root.resolve(),
                output,
                ucp_classes=cast(list[str], ucp_metadata["expected_classes"]),
                external_classes=cast(list[str], external_metadata["expected_classes"]),
                source_metadata={ucp_id: ucp_metadata, external_id: external_metadata},
                max_hamming=max_hamming,
            ),
            indent=2,
        )
    )


@app.command("plan-ablation")
def plan_ablation(manifest: Path, output: Path) -> None:
    """Create the registered nested ablation matrix from accepted training records."""
    samples: list[str] = []
    labels: list[int] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["status"] == "accepted" and record["split"] == "train":
            for box_index, box in enumerate(record["boxes"]):
                samples.append(f"{record['sample_id']}:{box_index}")
                labels.append(int(box["class_id"]))
    levels = build_nested_levels(samples, labels)
    payload = [
        {
            "requested_per_class": level.requested_per_class,
            "status": level.status,
            "reason": level.reason,
            "sample_ids": list(level.sample_ids),
            "subset_sha256": subset_digest(level),
        }
        for level in levels
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@app.command("plan-classification-ablation")
def plan_classification_ablation(manifest: Path, output: Path) -> None:
    """Plan nested UCP image-level subsets and the full-data condition."""
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    training = [row for row in rows if row.get("partition") == "internal" and row.get("split") == "train"]
    samples = [str(row["sample_id"]) for row in training]
    labels = [int(row["class_id"]) for row in training]
    levels = append_full_level(build_nested_levels(samples, labels), samples)
    payload = [
        {
            "requested_per_class": level.requested_per_class,
            "status": level.status,
            "reason": level.reason,
            "sample_ids": list(level.sample_ids),
            "subset_sha256": subset_digest(level),
        }
        for level in levels
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@app.command("audit-classification-runs")
def audit_classification_runs_command(plan: Path, runs_root: Path, output: Path) -> None:
    """Write a generated status ledger for every registered classification run."""
    typer.echo(audit_classification_run_matrix(plan, runs_root, output))


@app.command("train-detector")
def train_detector_command(config: Path) -> None:
    """Train one detector from a versioned YAML config."""
    typer.echo(train_detector(config))


@app.command("train-classifier")
def train_classifier_command(
    config: Path,
    data_root: Path,
    plan: Path,
    samples_per_class: str,
    output: Path,
    augmentation: bool = False,
) -> None:
    """Run one registered EfficientNet ablation condition."""
    level: int | str = samples_per_class if samples_per_class == "full" else int(samples_per_class)
    try:
        typer.echo(train_classifier(config, data_root, plan, level, augmentation, output))
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = output / "failure.json"
        failure.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "samples_per_class": level,
                    "augmentation": augmentation,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


@app.command("select-classifier")
def select_classifier_command(no_augmentation: Path, augmentation: Path, output: Path) -> None:
    """Select the production full-data condition from validation evidence."""
    typer.echo(select_classifier(no_augmentation, augmentation, output))


@app.command("evaluate-external")
def evaluate_external_command(checkpoint: Path, external_root: Path, output: Path) -> None:
    """Evaluate the selected classifier on isolated Abduls crops."""
    typer.echo(evaluate_external(checkpoint, external_root, output))


@app.command("export-classifier")
def export_classifier_command(checkpoint: Path, output: Path) -> None:
    """Export the selected classifier to ONNX."""
    typer.echo(export_classifier(checkpoint, output))


@app.command("validate-classifier-onnx")
def validate_classifier_onnx_command(checkpoint: Path, model: Path, test_root: Path, output: Path) -> None:
    """Compare ONNX and PyTorch outputs on the fixed internal test set."""
    typer.echo(validate_onnx_classifier(checkpoint, model, test_root, output))


@app.command("generate-gradcam")
def generate_gradcam_command(
    checkpoint: Path,
    test_root: Path,
    predictions: Path,
    output: Path,
    limit: int = 28,
) -> None:
    """Generate hash-bound post-evaluation Grad-CAM diagnostics."""
    typer.echo(generate_gradcam_report(checkpoint, test_root, predictions, output, limit))


@app.command("export-detector")
def export_detector_command(checkpoint: Path, output: Path) -> None:
    """Export a detector checkpoint to ONNX."""
    typer.echo(export_detector(checkpoint, output))


@app.command("evaluate-detector")
def evaluate_detector_command(checkpoint: Path, data_yaml: Path, output: Path) -> None:
    """Evaluate a checkpoint on the canonical test split."""
    typer.echo(evaluate_detector(checkpoint, data_yaml, output))


@app.command("choose-detector")
def choose_detector_command(nano_result: Path, small_result: Path, output: Path) -> None:
    """Apply the pre-declared nano-versus-small Pareto rule."""
    typer.echo(choose_detector(nano_result, small_result, output))


@app.command("export-embedder")
def export_embedder_command(checkpoint: Path, output: Path) -> None:
    """Export a trained EfficientNet backbone to ONNX."""
    typer.echo(export_embedder(checkpoint, output))


@app.command("extract-embeddings")
def extract_embeddings_command(model: Path, data_root: Path, output: Path) -> None:
    """Extract embeddings for one train, validation, or test crop tree."""
    typer.echo(extract_embeddings(model, data_root, output))


@app.command("fit-anomaly")
def fit_anomaly_command(train: Path, validation: Path, output: Path) -> None:
    """Fit and validation-calibrate the conditional k-NN index."""
    typer.echo(fit_anomaly_index(train, validation, output))


@app.command("evaluate-anomaly")
def evaluate_anomaly_command(embedder: Path, index: Path, test_root: Path, output: Path) -> None:
    """Run the synthetic-corruption proxy evaluation."""
    typer.echo(evaluate_anomaly_proxy(embedder, index, test_root, output))


@app.command("benchmark-detector")
def benchmark_detector_command(
    model: Path,
    image: Annotated[list[Path], typer.Option(help="Repeat for each benchmark image")],
    output: Path,
) -> None:
    """Benchmark direct ONNX CPU inference with recorded samples."""
    typer.echo(benchmark_detector(model, image, output))


@app.command("benchmark-classifier")
def benchmark_classifier_command(
    model: Path,
    temperature: float,
    image: Annotated[list[Path], typer.Option(help="Repeat for each benchmark image")],
    output: Path,
) -> None:
    """Benchmark checksum-bound ONNX classification on the current CPU."""
    typer.echo(benchmark_classifier(model, temperature, image, output))


@app.command("build-release")
def build_release_command(
    directory: Path,
    classifier_version: str,
    dataset_manifest_sha256: str,
    internal_result: Path,
    external_result: Path,
) -> None:
    """Create a quality-gated classification serving manifest."""
    typer.echo(
        build_release_manifest(
            directory,
            classifier_version,
            dataset_manifest_sha256,
            internal_result,
            external_result,
        )
    )


@app.command("validate-results")
def validate_results(
    path: Annotated[Path, typer.Argument()] = Path("artifacts/results"),
) -> None:
    """Reject untraceable or fabricated experiment records."""
    results = validate_result_directory(path)
    typer.echo(f"validated {len(results)} result records")
