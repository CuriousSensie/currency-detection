"""Audit and normalize extracted YOLO-format source datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from pkrvision.constants import DENOMINATION_TO_CLASS
from pkrvision.data.hashing import difference_hash, sha256_file
from pkrvision.data.schema import AuditSummary, Box, SampleRecord
from pkrvision.data.split import assert_no_cluster_leakage, assign_splits, cluster_duplicates

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _read_names(root: Path) -> list[str]:
    candidates = sorted(root.rglob("data.yaml"))
    if not candidates:
        raise ValueError(f"No data.yaml found below {root}.")
    data = yaml.safe_load(candidates[0].read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(value) for value in names]
    raise ValueError("data.yaml must contain a list or integer-keyed mapping named 'names'.")


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        reverse_index = parts[::-1].index("images")
        index = len(parts) - reverse_index - 1
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _parse_boxes(label_path: Path, class_map: dict[int, int]) -> list[Box]:
    if not label_path.is_file():
        raise ValueError("missing_label")
    boxes: list[Box] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"invalid_label_columns:{line_number}")
        source_class = int(fields[0])
        if source_class not in class_map:
            raise ValueError(f"unknown_source_class:{source_class}")
        boxes.append(
            Box(
                class_id=class_map[source_class],
                x_center=float(fields[1]),
                y_center=float(fields[2]),
                width=float(fields[3]),
                height=float(fields[4]),
            )
        )
    if not boxes:
        raise ValueError("empty_label")
    return boxes


def _normalized_class_schema(names: list[str]) -> list[str]:
    """Reduce supported label spellings to their denomination-bearing schema."""
    normalized: list[str] = []
    for name in names:
        digits = "".join(character for character in name if character.isdigit())
        normalized.append(digits or name.strip().upper())
    return normalized


def audit_source(
    root: Path,
    source_id: str,
    expected_classes: list[str] | None = None,
) -> tuple[list[SampleRecord], AuditSummary]:
    """Audit one extracted YOLO source and return canonical records plus reconciliation."""
    names = _read_names(root)
    if expected_classes is not None:
        actual_schema = _normalized_class_schema(names)
        expected_schema = _normalized_class_schema(expected_classes)
        if actual_schema != expected_schema:
            raise ValueError(
                f"{source_id} class schema changed: expected {expected_schema}, found {actual_schema}. "
                "Update the source declaration only after a manual audit."
            )
    normalized = [name.strip().upper().removeprefix("PKR_").removesuffix("RUPEE") for name in names]
    source_denominations: dict[int, int] = {}
    excluded_source_classes: set[int] = set()
    for index, name in enumerate(normalized):
        numeric = "".join(character for character in name if character.isdigit())
        if numeric == "75":
            excluded_source_classes.add(index)
        elif numeric and int(numeric) in DENOMINATION_TO_CLASS:
            source_denominations[index] = DENOMINATION_TO_CLASS[int(numeric)]
    if set(source_denominations.values()) != set(DENOMINATION_TO_CLASS.values()):
        raise ValueError(f"{source_id} does not expose exactly the seven required denominations: {names}")
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    records: list[SampleRecord] = []
    class_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for image_path in images:
        relative = image_path.relative_to(root).as_posix()
        image_sha = sha256_file(image_path)
        sample_id = hashlib.sha256(f"{source_id}\0{relative}\0{image_sha}".encode()).hexdigest()[:24]
        try:
            with Image.open(image_path) as source:
                source.load()
                width, height = source.size
                perceptual = difference_hash(source)
            raw_classes = []
            label_path = _label_path(image_path)
            if label_path.is_file():
                for raw in label_path.read_text(encoding="utf-8").splitlines():
                    if raw.strip():
                        raw_classes.append(int(raw.split()[0]))
            if any(value in excluded_source_classes for value in raw_classes):
                reasons["contains_excluded_pkr_75"] += 1
                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        source_id=source_id,
                        source_relative_path=relative,
                        image_sha256=image_sha,
                        perceptual_hash=perceptual,
                        width=width,
                        height=height,
                        boxes=[],
                        status="excluded",
                        reason="contains_excluded_pkr_75",
                    )
                )
                continue
            boxes = _parse_boxes(label_path, source_denominations)
            for box in boxes:
                class_counts[str(box.class_id)] += 1
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    source_id=source_id,
                    source_relative_path=relative,
                    image_sha256=image_sha,
                    perceptual_hash=perceptual,
                    width=width,
                    height=height,
                    boxes=boxes,
                    status="accepted",
                )
            )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            reason = str(exc).split(":", 1)[0] or "invalid_image_or_label"
            reasons[reason] += 1
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    source_id=source_id,
                    source_relative_path=relative,
                    image_sha256=image_sha,
                    perceptual_hash="0" * 16,
                    width=1,
                    height=1,
                    boxes=[],
                    status="invalid",
                    reason=str(exc),
                )
            )
    return records, AuditSummary(
        source_id=source_id,
        total_images=len(records),
        accepted_images=sum(record.status == "accepted" for record in records),
        excluded_images=sum(record.status == "excluded" for record in records),
        invalid_images=sum(record.status == "invalid" for record in records),
        class_counts=dict(class_counts),
        reasons=dict(reasons),
    )


def prepare_dataset(
    sources: dict[str, Path],
    output: Path,
    *,
    source_metadata: dict[str, dict[str, object]] | None = None,
    max_hamming: int = 4,
) -> dict[str, object]:
    """Audit, merge, de-duplicate, split, and materialize canonical YOLO data."""
    all_records: list[SampleRecord] = []
    summaries: list[AuditSummary] = []
    for source_id, root in sorted(sources.items()):
        expected_classes: list[str] | None = None
        if source_metadata and source_id in source_metadata:
            configured = source_metadata[source_id].get("expected_classes")
            if isinstance(configured, list) and all(isinstance(item, str) for item in configured):
                expected_classes = configured
        records, summary = audit_source(root, source_id, expected_classes)
        all_records.extend(records)
        summaries.append(summary)
    cluster_duplicates(all_records, max_hamming=max_hamming)
    assign_splits(all_records)
    assert_no_cluster_leakage(all_records)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(record.model_dump_json() + "\n")
    for record in all_records:
        if record.status != "accepted" or record.split is None:
            continue
        source_path = sources[record.source_id] / record.source_relative_path
        extension = source_path.suffix.lower()
        destination = output / "yolo" / record.split / "images" / f"{record.sample_id}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        label_path = output / "yolo" / record.split / "labels" / f"{record.sample_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(
                f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} {box.width:.8f} {box.height:.8f}"
                for box in record.boxes
            )
            + "\n",
            encoding="utf-8",
        )
        with Image.open(source_path) as crop_source:
            rgb = crop_source.convert("RGB")
            for box_index, box in enumerate(record.boxes):
                left = max(0, round((box.x_center - box.width / 2) * rgb.width))
                top = max(0, round((box.y_center - box.height / 2) * rgb.height))
                right = min(rgb.width, round((box.x_center + box.width / 2) * rgb.width))
                bottom = min(rgb.height, round((box.y_center + box.height / 2) * rgb.height))
                if right <= left or bottom <= top:
                    continue
                denomination = (10, 20, 50, 100, 500, 1000, 5000)[box.class_id]
                crop_path = (
                    output / "classification" / record.split / str(denomination) / f"{record.sample_id}-{box_index}.jpg"
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                rgb.crop((left, top, right, bottom)).save(crop_path, quality=95)
    data_yaml = {
        # Omitting ``path`` makes Ultralytics use this YAML's parent directory.
        # Keeping split paths relative makes the canonical bundle portable to Colab.
        "train": "yolo/train/images",
        "val": "yolo/validation/images",
        "test": "yolo/test/images",
        "names": {index: str(value) for index, value in enumerate((10, 20, 50, 100, 500, 1000, 5000))},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    contact_cells: list[tuple[int, Path]] = []
    for denomination in (10, 20, 50, 100, 500, 1000, 5000):
        candidates = sorted((output / "classification" / "train" / str(denomination)).glob("*.jpg"))
        contact_cells.extend((denomination, path) for path in candidates[:4])
    if contact_cells:
        thumb_size = (240, 140)
        sheet = Image.new("RGB", (thumb_size[0] * 4, thumb_size[1] * 7), "white")
        draw = ImageDraw.Draw(sheet)
        for position, (denomination, path) in enumerate(contact_cells):
            with Image.open(path) as crop:
                thumb = ImageOps.contain(crop.convert("RGB"), (220, 112))
            column = position % 4
            row = position // 4
            left = column * thumb_size[0] + (thumb_size[0] - thumb.width) // 2
            top = row * thumb_size[1] + 8
            sheet.paste(thumb, (left, top))
            draw.text((column * thumb_size[0] + 10, row * thumb_size[1] + 122), f"PKR {denomination}", fill="black")
        sheet.save(output / "contact-sheet.jpg", quality=92)
    split_class_counts: dict[str, Counter[str]] = {split: Counter() for split in ("train", "validation", "test")}
    for record in all_records:
        if record.status == "accepted" and record.split is not None:
            for box in record.boxes:
                split_class_counts[record.split][str(box.class_id)] += 1
    summary_payload: dict[str, object] = {
        "status": "complete",
        "sources": [summary.model_dump() for summary in summaries],
        "source_metadata": source_metadata or {},
        "accepted": sum(record.status == "accepted" for record in all_records),
        "excluded": sum(record.status == "excluded" for record in all_records),
        "invalid": sum(record.status == "invalid" for record in all_records),
        "split_counts": dict(Counter(record.split for record in all_records if record.split)),
        "class_counts_by_split": {split: dict(counts) for split, counts in split_class_counts.items()},
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    return summary_payload
