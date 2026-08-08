"""Prepare leakage-resistant internal and cross-dataset classification data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from pkrvision.constants import CLASS_TO_DENOMINATION
from pkrvision.data.audit import audit_source
from pkrvision.data.hashing import difference_hash, sha256_file
from pkrvision.data.schema import SampleRecord
from pkrvision.data.split import assert_no_cluster_leakage, assign_splits, cluster_duplicates
from pkrvision.preprocessing import letterbox


def _crop_box(image: Image.Image, record: SampleRecord, box_index: int) -> Image.Image:
    box = record.boxes[box_index]
    left = max(0, round((box.x_center - box.width / 2) * image.width))
    top = max(0, round((box.y_center - box.height / 2) * image.height))
    right = min(image.width, round((box.x_center + box.width / 2) * image.width))
    bottom = min(image.height, round((box.y_center + box.height / 2) * image.height))
    if right <= left or bottom <= top:
        raise ValueError("invalid_crop_bounds")
    return image.crop((left, top, right, bottom))


def _save_derived(image: Image.Image, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    letterbox(image).save(destination, format="JPEG", quality=95, optimize=False, progressive=False)
    return sha256_file(destination)


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def prepare_classification_dataset(
    ucp_root: Path,
    external_root: Path,
    output: Path,
    *,
    ucp_classes: list[str],
    external_classes: list[str],
    source_metadata: dict[str, object],
    max_hamming: int = 4,
) -> dict[str, object]:
    """Create UCP image-level splits and an isolated Abduls crop evaluation set."""
    if output.exists():
        raise FileExistsError(output)
    ucp_records, ucp_summary = audit_source(ucp_root, "ucp-pakistani-currency-note-data", ucp_classes)
    eligible: list[SampleRecord] = []
    ambiguous = 0
    for record in ucp_records:
        if record.status != "accepted":
            continue
        classes = {box.class_id for box in record.boxes}
        if len(classes) != 1:
            record.status = "invalid"
            record.reason = "ambiguous_image_level_denomination"
            ambiguous += 1
            continue
        eligible.append(record)
    cluster_duplicates(eligible, max_hamming=max_hamming)
    assign_splits(eligible, one_label_per_record=True)
    assert_no_cluster_leakage(eligible)

    manifest_rows: list[dict[str, object]] = []
    internal_hashes: list[str] = []
    split_counts: Counter[str] = Counter()
    split_class_counts: dict[str, Counter[str]] = {split: Counter() for split in ("train", "validation", "test")}
    for record in eligible:
        assert record.split is not None
        class_id = record.boxes[0].class_id
        denomination = CLASS_TO_DENOMINATION[class_id]
        source = ucp_root / record.source_relative_path
        destination = output / "internal" / record.split / str(denomination) / f"{record.sample_id}.jpg"
        with Image.open(source) as image:
            derived_hash = _save_derived(image, destination)
            normalized_dhash = difference_hash(letterbox(image))
        internal_hashes.append(normalized_dhash)
        split_counts[record.split] += 1
        split_class_counts[record.split][str(denomination)] += 1
        manifest_rows.append(
            {
                "sample_id": record.sample_id,
                "source_id": record.source_id,
                "source_relative_path": record.source_relative_path,
                "source_sha256": record.image_sha256,
                "derived_relative_path": destination.relative_to(output).as_posix(),
                "derived_sha256": derived_hash,
                "perceptual_hash": normalized_dhash,
                "duplicate_cluster_id": record.duplicate_cluster_id,
                "denomination_pkr": denomination,
                "class_id": class_id,
                "partition": "internal",
                "split": record.split,
                "label_derivation": "unanimous-source-annotations",
            }
        )

    external_records, external_summary = audit_source(external_root, "abduls-pkr-notes", external_classes)
    external_counts: Counter[str] = Counter()
    external_excluded_overlap = 0
    external_invalid_crops = 0
    for record in external_records:
        if record.status != "accepted":
            continue
        source = external_root / record.source_relative_path
        with Image.open(source) as opened:
            external_image = opened.convert("RGB")
            for box_index, box in enumerate(record.boxes):
                denomination = CLASS_TO_DENOMINATION[box.class_id]
                try:
                    crop = _crop_box(external_image, record, box_index)
                except ValueError:
                    external_invalid_crops += 1
                    continue
                normalized = letterbox(crop)
                perceptual = difference_hash(normalized)
                if any(_hamming(perceptual, candidate) <= max_hamming for candidate in internal_hashes):
                    external_excluded_overlap += 1
                    continue
                sample_id = hashlib.sha256(f"{record.sample_id}:{box_index}".encode()).hexdigest()[:24]
                destination = output / "external" / str(denomination) / f"{sample_id}.jpg"
                derived_hash = _save_derived(crop, destination)
                external_counts[str(denomination)] += 1
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_id": record.source_id,
                        "source_relative_path": record.source_relative_path,
                        "source_sha256": record.image_sha256,
                        "derived_relative_path": destination.relative_to(output).as_posix(),
                        "derived_sha256": derived_hash,
                        "perceptual_hash": perceptual,
                        "duplicate_cluster_id": record.duplicate_cluster_id,
                        "denomination_pkr": denomination,
                        "class_id": box.class_id,
                        "partition": "external",
                        "split": "external",
                        "box_index": box_index,
                        "bbox_xywh_normalized": [box.x_center, box.y_center, box.width, box.height],
                        "label_derivation": "ground-truth-whole-note-box",
                    }
                )
                crop.close()

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "samples.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8")
    summary: dict[str, object] = {
        "status": "complete",
        "task": "single-banknote-image-classification",
        "source_metadata": source_metadata,
        "internal": {
            "source_audit": ucp_summary.model_dump(),
            "ambiguous_images": ambiguous,
            "accepted_images": len(eligible),
            "split_counts": dict(split_counts),
            "class_counts_by_split": {split: dict(counts) for split, counts in split_class_counts.items()},
        },
        "external": {
            "source_audit": external_summary.model_dump(),
            "accepted_crops": sum(external_counts.values()),
            "class_counts": dict(external_counts),
            "excluded_cross_source_overlap": external_excluded_overlap,
            "invalid_crops": external_invalid_crops,
        },
        "manifest_sha256": sha256_file(manifest),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
