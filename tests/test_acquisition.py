import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image
from pytest import MonkeyPatch

from pkrvision.data.acquisition import download_roboflow, extract_archive


class FakeRoboflow:
    """Minimal SDK double that verifies the target does not pre-exist."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def workspace(self, _workspace: str) -> "FakeRoboflow":
        return self

    def project(self, _project: str) -> "FakeRoboflow":
        return self

    def version(self, _version: int) -> "FakeRoboflow":
        return self

    def download(self, _format: str, location: str, overwrite: bool) -> object:
        destination = Path(location)
        assert not destination.exists()
        assert overwrite is False
        (destination / "train" / "images").mkdir(parents=True)
        (destination / "data.yaml").write_text("names: ['10']\n", encoding="utf-8")
        Image.new("RGB", (8, 8), "green").save(destination / "train" / "images" / "sample.jpg")

        class Downloaded:
            location = str(destination)

        return Downloaded()


def test_download_does_not_precreate_sdk_destination(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    module = ModuleType("roboflow")
    module.Roboflow = FakeRoboflow  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roboflow", module)
    destination = tmp_path / "source"
    record = download_roboflow("workspace", "project", 2, "secret", destination)
    assert record["image_count"] == "1"
    assert len(record["extracted_tree_sha256"]) == 64
    assert (destination / "acquisition.json").is_file()


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe path"):
        extract_archive(archive, tmp_path / "destination")
