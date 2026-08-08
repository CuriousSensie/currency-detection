"""Explicit, checksum-recorded dataset acquisition."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pkrvision.data.hashing import sha256_file


def directory_sha256(root: Path) -> str:
    """Hash extracted source contents including relative paths and file bytes."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "acquisition.json")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, destination: Path, expected_sha256: str | None = None) -> dict[str, str]:
    """Verify and safely extract a manually downloaded ZIP archive."""
    actual = sha256_file(archive)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"Archive checksum mismatch: expected {expected_sha256}, received {actual}.")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            portable_name = member.filename.replace("\\", "/")
            member_path = PurePosixPath(portable_name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Archive contains an unsafe path: {member.filename}")
        bundle.extractall(destination)
    record = {
        "archive": archive.name,
        "archive_sha256": actual,
        "extracted_at": datetime.now(UTC).isoformat(),
        "destination": str(destination),
    }
    (destination / "acquisition.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def download_roboflow(
    workspace: str, project_name: str, version: int, api_key: str, destination: Path
) -> dict[str, str]:
    """Download one explicit Roboflow version using the authenticated SDK."""
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY is required and must not be committed.")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("Install training dependencies to use authenticated acquisition.") from exc
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = Roboflow(api_key=api_key).workspace(workspace).project(project_name).version(version)
    downloaded = dataset.download("yolov8", location=str(destination), overwrite=False)
    image_count = sum(1 for path in destination.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if image_count == 0 or not any(destination.rglob("data.yaml")):
        raise RuntimeError(f"Roboflow export completed without a usable YOLO dataset at {destination}.")
    record = {
        "workspace": workspace,
        "project": project_name,
        "version": str(version),
        "format": "yolov8",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "destination": str(Path(downloaded.location).resolve()),
        "extracted_tree_sha256": directory_sha256(destination),
        "image_count": str(image_count),
    }
    (destination / "acquisition.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def copy_manual_directory(source: Path, destination: Path) -> None:
    """Copy an already extracted source into immutable raw-data storage."""
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
