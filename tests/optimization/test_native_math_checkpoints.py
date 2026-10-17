"""Opt-in real GPU gates; each checkpoint executes in supervised fresh workers."""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MLX_ONE_RUN_P01_CHECKPOINTS") != "1",
                               reason="opt-in supervised native P0.1 checkpoint qualification")

CASES = [
    ("qwen3", "mlx-community/Qwen3-0.6B-8bit", "reference", "reference"),
    ("qwen2", "mlx-community/Qwen2.5-0.5B-Instruct-8bit", "reference", "reference"),
    ("lfm2", "mlx-community/LFM2-350M-4bit", "reference", "reference"),
    ("gpt2", "openai-community/gpt2", "reference", "reference"),
    ("qwen35", "mlx-community/Qwen3.5-0.8B-4bit", "reference", "reference"),
    *[(name, model, attention, projection)
      for name, model in (("qwen3", "mlx-community/Qwen3-0.6B-8bit"),
                          ("qwen2", "mlx-community/Qwen2.5-0.5B-Instruct-8bit"))
      for attention, projection in (("reference", "mlx_qmm"),)],
]


@pytest.mark.parametrize("name,checkpoint,attention,projection", CASES,
                         ids=[f"{n}-{a}-{p}" for n, _, a, p in CASES])
def test_native_last_logits_and_shared_operators(name, checkpoint, attention, projection):
    from qualification.p01_experiments import main

    baseline = os.environ.get("MLX_ONE_P01_BASELINE")
    if not baseline or not (Path(baseline) / "download_manifest.json").is_file():
        pytest.skip("immutable completed native asset/environment manifest not supplied")
    output = Path("qualification/results/p01_math/checkpoints") / f"{name}-{attention}-{projection}"
    result = main(["--baseline", baseline, "--model", checkpoint, "--route", "optimized",
        "--workload", "parity", "--prompt-tokens", "128", "--repetitions", "1",
        "--attention", attention, "--projection", projection, "--output", str(output), "--run"])
    assert result == 0
    manifest_path = max(output.glob("*/manifest.json"), key=lambda p: p.parent.name)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["tasks"]) == 2
    assert all(task["status"] == "PASS" for task in manifest["tasks"])
    assert manifest["qualification_promoted"] is False
