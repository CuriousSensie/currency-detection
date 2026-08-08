"""Deterministic duplicate-aware dataset splitting."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from pkrvision.constants import PROJECT_SEED
from pkrvision.data.hashing import hamming_distance
from pkrvision.data.schema import SampleRecord


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_duplicates(records: list[SampleRecord], max_hamming: int = 4) -> None:
    """Assign transitive exact and near-duplicate clusters in place."""
    accepted = [record for record in records if record.status == "accepted"]
    union = _UnionFind([record.sample_id for record in accepted])
    by_sha: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in accepted:
        by_sha[record.image_sha256].append(record)
    for group in by_sha.values():
        for record in group[1:]:
            union.union(group[0].sample_id, record.sample_id)
    # Dataset scale is small (~5k). Pairwise dHash comparison is transparent and deterministic.
    for position, left in enumerate(accepted):
        for right in accepted[position + 1 :]:
            if hamming_distance(left.perceptual_hash, right.perceptual_hash) <= max_hamming:
                union.union(left.sample_id, right.sample_id)
    roots = sorted({union.find(record.sample_id) for record in accepted})
    root_ids = {root: f"cluster-{index:06d}" for index, root in enumerate(roots)}
    for record in accepted:
        record.duplicate_cluster_id = root_ids[union.find(record.sample_id)]


def assign_splits(
    records: list[SampleRecord],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = PROJECT_SEED,
    *,
    one_label_per_record: bool = False,
) -> None:
    """Greedily assign duplicate groups while balancing sizes and class counts."""
    if abs(sum(ratios) - 1.0) > 1e-9 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must be positive and sum to one.")
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.status == "accepted":
            if record.duplicate_cluster_id is None:
                raise ValueError("Duplicate clustering must run before split assignment.")
            groups[record.duplicate_cluster_id].append(record)
    rng = random.Random(seed)
    ranked = list(groups.items())
    rng.shuffle(ranked)
    ranked.sort(key=lambda item: len(item[1]), reverse=True)
    names = ("train", "validation", "test")
    target_sizes = {
        name: len([r for r in records if r.status == "accepted"]) * ratio
        for name, ratio in zip(names, ratios, strict=True)
    }
    total_classes: Counter[int] = Counter(
        record.boxes[0].class_id
        for record in records
        if record.status == "accepted" and one_label_per_record
    )
    if not one_label_per_record:
        total_classes = Counter(
            box.class_id for record in records if record.status == "accepted" for box in record.boxes
        )
    target_classes = {
        name: {class_id: count * ratio for class_id, count in total_classes.items()}
        for name, ratio in zip(names, ratios, strict=True)
    }
    current_sizes: Counter[str] = Counter()
    current_classes: dict[str, Counter[int]] = {name: Counter() for name in names}
    for _, group in ranked:
        group_classes = (
            Counter(record.boxes[0].class_id for record in group)
            if one_label_per_record
            else Counter(box.class_id for record in group for box in record.boxes)
        )

        def cost(
            name: str,
            group_size: int = len(group),
            class_counts: Counter[int] = group_classes,
        ) -> float:
            current_size_error = (current_sizes[name] - target_sizes[name]) ** 2 / max(target_sizes[name], 1)
            next_size_error = ((current_sizes[name] + group_size) - target_sizes[name]) ** 2 / max(
                target_sizes[name], 1
            )
            size_cost = next_size_error - current_size_error
            class_cost = sum(
                (
                    ((current_classes[name][class_id] + count) - target_classes[name][class_id]) ** 2
                    - (current_classes[name][class_id] - target_classes[name][class_id]) ** 2
                )
                / max(target_classes[name][class_id], 1)
                for class_id, count in class_counts.items()
            )
            return size_cost + class_cost

        chosen = min(names, key=lambda name: (cost(name), current_sizes[name], name))
        for record in group:
            record.split = chosen  # type: ignore[assignment]
        current_sizes[chosen] += len(group)
        current_classes[chosen].update(group_classes)


def assert_no_cluster_leakage(records: list[SampleRecord]) -> None:
    """Raise when one duplicate cluster appears in multiple splits."""
    seen: dict[str, str] = {}
    for record in records:
        if record.status != "accepted":
            continue
        assert record.duplicate_cluster_id is not None and record.split is not None
        existing = seen.setdefault(record.duplicate_cluster_id, record.split)
        if existing != record.split:
            raise AssertionError(f"Duplicate cluster {record.duplicate_cluster_id} crosses splits.")
