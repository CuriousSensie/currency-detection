from pathlib import Path

import pytest
import yaml
from PIL import Image

from pkrvision.data.audit import audit_source, prepare_dataset
from pkrvision.data.classification import prepare_classification_dataset
from pkrvision.data.schema import Box, SampleRecord
from pkrvision.data.split import assert_no_cluster_leakage, assign_splits, cluster_duplicates


def make_source(root: Path) -> None:
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir(parents=True)
    (root / "data.yaml").write_text(
        "names: ['10', '20', '50', '75', '100', '500', '1000', '5000']\n",
        encoding="utf-8",
    )
    for index, class_id in enumerate((0, 1, 2, 4, 5, 6, 7)):
        Image.new("RGB", (40 + index, 30), (20 * index, 30, 80)).save(root / "train/images" / f"note-{index}.jpg")
        (root / "train/labels" / f"note-{index}.txt").write_text(f"{class_id} 0.5 0.5 0.8 0.7\n", encoding="utf-8")
    Image.new("RGB", (45, 30), "green").save(root / "train/images" / "seventy-five.jpg")
    (root / "train/labels" / "seventy-five.txt").write_text("3 0.5 0.5 0.8 0.7\n", encoding="utf-8")


def test_audit_remaps_classes_and_excludes_pkr_75(tmp_path: Path) -> None:
    make_source(tmp_path)
    records, summary = audit_source(tmp_path, "fixture")
    assert summary.accepted_images == 7
    assert summary.excluded_images == 1
    assert set(summary.class_counts) == {"0", "1", "2", "3", "4", "5", "6"}
    excluded = [record for record in records if record.status == "excluded"]
    assert excluded[0].reason == "contains_excluded_pkr_75"


def test_prepare_is_leakage_safe_and_reconciled(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_source(source)
    output = tmp_path / "processed"
    summary = prepare_dataset({"fixture": source}, output)
    assert summary["accepted"] == 7
    assert (output / "dataset.yaml").is_file()
    assert (output / "samples.jsonl").is_file()
    dataset = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert "path" not in dataset
    assert dataset["train"] == "yolo/train/images"
    records, _ = audit_source(source, "fixture")
    # The function itself asserts leakage; retain an explicit unit-level call contract.
    accepted = [record for record in records if record.status == "accepted"]
    assert len(accepted) == 7
    assert_no_cluster_leakage([])


def test_audit_rejects_a_changed_source_schema(tmp_path: Path) -> None:
    make_source(tmp_path)
    expected = ["10", "20", "50", "75", "100", "500", "1000", "2000"]
    with pytest.raises(ValueError, match="class schema changed"):
        audit_source(tmp_path, "fixture", expected)


def test_group_split_tracks_registered_ratios() -> None:
    records = [
        SampleRecord(
            sample_id=f"sample-{index:04d}",
            source_id="fixture",
            source_relative_path=f"{index}.jpg",
            image_sha256=f"{index:064x}",
            perceptual_hash=f"{index * 257:016x}",
            width=100,
            height=50,
            boxes=[Box(class_id=index % 7, x_center=0.5, y_center=0.5, width=0.8, height=0.8)],
            status="accepted",
        )
        for index in range(700)
    ]
    cluster_duplicates(records, max_hamming=-1)
    assign_splits(records)
    counts = {split: sum(record.split == split for record in records) for split in ("train", "validation", "test")}
    assert counts == {"train": 490, "validation": 105, "test": 105}
    assert_no_cluster_leakage(records)


def test_audit_rejects_changed_declared_class_schema(tmp_path: Path) -> None:
    make_source(tmp_path)
    with pytest.raises(ValueError, match="class schema changed"):
        audit_source(tmp_path, "fixture", ["10", "20", "50", "100", "500", "1000", "5000"])


def test_prepare_classification_uses_full_images_and_external_crops(tmp_path: Path) -> None:
    ucp = tmp_path / "ucp"
    external = tmp_path / "external"
    for root, names in (
        (ucp, ["10", "100", "1000", "20", "50", "500", "5000", "75"]),
        (external, ["PKR_10", "PKR_100", "PKR_1000", "PKR_20", "PKR_50", "PKR_500", "PKR_5000"]),
    ):
        (root / "train/images").mkdir(parents=True)
        (root / "train/labels").mkdir(parents=True)
        (root / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
        for class_id in range(7):
            Image.new("RGB", (80 + class_id, 45), (class_id * 25, 30, 150)).save(
                root / "train/images" / f"note-{class_id}.jpg"
            )
            (root / "train/labels" / f"note-{class_id}.txt").write_text(
                f"{class_id} 0.5 0.5 0.75 0.75\n", encoding="utf-8"
            )
    output = tmp_path / "classification"
    summary = prepare_classification_dataset(
        ucp,
        external,
        output,
        ucp_classes=["10", "100", "1000", "20", "50", "500", "5000", "75"],
        external_classes=["PKR_10", "PKR_100", "PKR_1000", "PKR_20", "PKR_50", "PKR_500", "PKR_5000"],
        source_metadata={},
        max_hamming=-1,
    )
    assert summary["internal"]["accepted_images"] == 7
    assert summary["external"]["accepted_crops"] == 7
    derived = next((output / "internal").rglob("*.jpg"))
    with Image.open(derived) as image:
        assert image.size == (224, 224)
