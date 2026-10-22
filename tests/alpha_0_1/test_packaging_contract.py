"""Audit install inputs/resources without packaging unapproved local source."""

import re
from importlib import resources
from pathlib import Path

import pytest

from mlx_one import __version__

ROOT = Path(__file__).resolve().parents[2]


def test_package_versions_and_explicit_cpu_runtime_dependency():
    # tomllib is unavailable on Python 3.10, which remains a declared target.
    from setuptools.config.pyprojecttoml import read_configuration

    config = read_configuration(ROOT / "pyproject.toml", expand=False)
    assert config["project"]["version"] == __version__
    dependencies = config["project"]["dependencies"]
    assert any(item.startswith("numpy") for item in dependencies)
    assert not any(item.startswith(("mlx-lm", "mlx-vlm", "mlx-audio", "mlx-embeddings"))
                   for item in dependencies)


def test_packaged_ui_entry_references_existing_local_assets():
    dist = resources.files("mlx_one").joinpath("ui/dist")
    html = dist.joinpath("index.html").read_text()
    paths = re.findall(r'(?:src|href)="([^"]+)"', html)
    checked = []
    for path in paths:
        if path.startswith(("http:", "https:", "data:", "#")) or path == "/":
            continue
        relative = path.removeprefix("./").lstrip("/").split("?")[0]
        if relative and "." in relative.rsplit("/", 1)[-1]:
            assert dist.joinpath(relative).is_file(), relative
            checked.append(relative)
    assert checked


@pytest.mark.parametrize("directory", ["compatibility", "hardware", "calibrations"])
def test_packaged_metadata_resources_are_present(directory):
    folder = resources.files("mlx_one").joinpath("data", directory)
    assert any(path.is_file() for path in folder.iterdir())
