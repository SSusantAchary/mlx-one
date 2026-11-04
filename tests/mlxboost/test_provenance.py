import ast
import hashlib
from pathlib import Path

import pytest

from mlx_one.inference_baseline import TOLERANCES
from mlx_one.mlxboost.provenance import source_manifest, verify_captures

ROOT = Path(__file__).resolve().parents[2]


def test_source_manifest_schema_and_sha_verification(tmp_path):
    manifest = source_manifest()
    assert {s["name"] for s in manifest["upstream_repositories"]} == {
        "mlx",
        "mlx_comparison",
        "mlx_lm",
        "omlx",
    }
    assert all(
        len(f["sha256"]) == 64 for s in manifest["upstream_repositories"] for f in s["files"]
    )
    with pytest.raises(FileNotFoundError):
        verify_captures(tmp_path)


def test_retained_captures_when_available():
    if not (ROOT / "local/mlxboost/sources/sources.json").is_file():
        pytest.skip("local source capture is development evidence")
    assert verify_captures(ROOT) >= 20


def test_no_reference_numerical_gate_relaxation():
    assert TOLERANCES == {
        "float32": (1e-4, 1e-4),
        "float16": (1e-2, 1e-2),
        "bfloat16": (5e-2, 1e-2),
    }


def test_no_upstream_runtime_imports_in_mlxboost():
    for path in (ROOT / "src/mlx_one/mlxboost").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [n.name.split(".")[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert not set(names) & {"mlx_lm", "omlx", "transformers"}


def test_custom_kernel_not_required_for_install_or_import():
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
class BanNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'omlx', 'transformers'}:
            raise AssertionError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BanNative())
from mlx_one.mlxboost import MLXBoostConfig
from mlx_one.mlxboost.provenance import source_manifest
assert not MLXBoostConfig().enabled
assert len(source_manifest()['upstream_repositories']) == 4
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_historical_evidence_hashes_when_available():
    import json

    audit = ROOT / "local/mlxboost/runs/audit-20261101/prechange/manifest.json"
    if not audit.exists():
        pytest.skip("local historical evidence is not distributed")
    directory = ROOT / "local/competitor-comparison/runs/comparison-20261101"
    for path, expected in json.loads(audit.read_text())["historical_sha256"].items():
        assert hashlib.sha256((directory / path).read_bytes()).hexdigest() == expected
