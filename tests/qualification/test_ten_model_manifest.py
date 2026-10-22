"""Exact cohort and cache resolution cannot silently substitute derivatives."""

import importlib.util
import json
from pathlib import Path


def load_runner():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "ten_model_runner", root / "scripts/qualification/qualify_ten_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module


def test_manifest_preserves_ten_exact_targets_and_eight_families():
    root, runner = load_runner()
    rows = runner.manifest(root / "scripts/qualification/ten_model_manifest.yaml")
    assert len(rows) == 10 and len({r["family"] for r in rows}) == 8
    assert rows[1]["repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert rows[7]["repo"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert rows[1]["family"] == rows[7]["family"] == "qwen2"
    assert rows[5]["family"] == rows[6]["family"] == "llama"


def test_derivative_cache_does_not_satisfy_upstream_and_partial_shards_are_missing(tmp_path):
    _, runner = load_runner()
    derivative = tmp_path / "models--mlx-community--Qwen3-0.6B-8bit/snapshots/revision"
    derivative.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (derivative / name).write_text("{}")
    assert runner.resolve_cached("Qwen/Qwen3-0.6B", tmp_path) is None
    exact = tmp_path / "models--Qwen--Qwen3-0.6B/snapshots/revision"
    exact.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model-1.safetensors"):
        (exact / name).write_text("{}")
    (exact / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    )
    assert runner.resolve_cached("Qwen/Qwen3-0.6B", tmp_path) is None
    (exact / "model-2.safetensors").write_text("{}")
    assert runner.resolve_cached("Qwen/Qwen3-0.6B", tmp_path) == exact
