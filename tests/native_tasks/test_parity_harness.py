from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = spec_from_file_location(name, ROOT / "benchmarks" / "native_tasks" / f"{name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("compare_tensors")
evidence = load("evidence")


def test_comparison_catches_divergence_shape_nonfinite_and_ranking():
    ref = np.array([[1.0, 2.0]], dtype=np.float32)
    assert compare.compare(ref, ref, atol=1e-4, rtol=1e-4)["passed"]
    changed = compare.compare(ref, np.array([[2.0, 1.0]]), atol=1e-4, rtol=1e-4)
    assert not changed["passed"] and not changed["top1_agreement"]
    assert compare.compare(ref, ref[0], atol=1e-4, rtol=1e-4)["reason"] == "shape_mismatch"
    assert (
        compare.compare(ref, np.array([[np.nan, 2.0]]), atol=1e-4, rtol=1e-4)["reason"]
        == "nonfinite"
    )


def test_archive_requires_exact_inventory(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, first=[1.0], last=[2.0])
    np.savez(b, first=[1.0], other=[2.0])
    report = compare.compare_archives(a, b, atol=0, rtol=0)
    assert not report["passed"] and report["missing"] == ["last"]
    assert report["unexpected"] == ["other"]


def test_manifest_requires_pins_and_never_overwrites(tmp_path):
    artifact = tmp_path / "trace"
    artifact.write_bytes(b"trace")
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        evidence.write_manifest(path, {}, {"trace": artifact})
    metadata = {"source_commit": "a" * 40, "model_revision": "b" * 40}
    evidence.write_manifest(path, metadata, {"trace": artifact})
    with pytest.raises(FileExistsError):
        evidence.write_manifest(path, metadata, {"trace": artifact})


def test_production_imports_do_not_load_reference_frameworks():
    import subprocess
    import sys

    script = """
import sys
import mlx_one.vision.vit_classification
import mlx_one.models.classification.distilbert
for name in ('torch', 'transformers', 'onnxruntime', 'mlx_lm', 'omlx'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_lockfiles_use_full_immutable_revisions():
    import re

    import yaml

    source = yaml.safe_load((ROOT / "docs/native_tasks/sources.lock.yaml").read_text())
    checkpoint = yaml.safe_load((ROOT / "docs/native_tasks/checkpoints.lock.yaml").read_text())
    full_sha = re.compile(r"[0-9a-f]{40}\Z")
    for entry in source.values():
        assert full_sha.fullmatch(entry["commit"])
    assert checkpoint["checkpoints"]
    for entry in checkpoint["checkpoints"]:
        assert full_sha.fullmatch(entry["revision"])
        if isinstance(entry["asset_hashes"], dict):
            assert all(
                re.fullmatch(r"[0-9a-f]{64}", digest) for digest in entry["asset_hashes"].values()
            )
