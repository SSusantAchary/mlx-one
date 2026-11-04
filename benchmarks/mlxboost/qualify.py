"""Forced-token checkpoint parity against the SHA-verified MLX-LM source tree.

Development/reference tooling only. Both models use the development MLX 0.32.2
backend; this is explicitly a matched-dependency numerical ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
REFERENCE = ROOT / "local/mlxboost/sources/mlx_lm/tree"
sys.path.insert(0, str(REFERENCE))
# The Git tree excludes the setuptools-scm generated wheel version file.
# Supply only that generated metadata, never substitute implementation modules.
version_module = ModuleType("mlx_lm._version")
version_module.__version__ = "0.32.2+source.93e6d4a"
sys.modules["mlx_lm._version"] = version_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen2", "qwen3", "smollm2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.text import load_text_model

    sources = json.loads((REFERENCE.parent.parent / "sources.json").read_text())
    pin = next(p for p in sources["upstream_repositories"] if p["name"] == "mlx_lm")
    for f in pin["files"]:
        if hashlib.sha256((REFERENCE / f["path"]).read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError("reference source changed: " + f["path"])
    models = json.loads((ROOT / "local/competitor-comparison/models.json").read_text())["models"]
    spec = next(m for m in models if m["key"] == args.model)
    from evidence import snapshot

    snapshot(args.output, model=spec)
    native = load_text_model(spec["path"], offline=True)
    configure_execution_dtype(native, spec["execution_dtype"])
    reference, _ = load_model(Path(spec["path"]))
    reference.set_dtype(
        getattr(mx, spec["execution_dtype"]),
        predicate=lambda d: d in {mx.float16, mx.bfloat16, mx.float32},
    )
    mx.eval(reference.parameters())
    source_cache = make_prompt_cache(reference)
    local_cache = native.model.make_cache()
    prompt = spec["prompts"][1]["token_ids"]
    current = prompt
    history, rows, tokens = list(prompt), [], []
    raw = {}

    def array(x):
        return np.array(x.astype(mx.float32))

    def compare(name, left, right, step):
        a, b = array(left), array(right)
        row = {"name": name, "step": step, **compare_tensors(a, b, dtype=spec["execution_dtype"])}
        rows.append(row)
        if step in {0, 15, args.steps - 1} and name == "logits":
            raw[f"source_logits_{step}"] = a
            raw[f"native_logits_{step}"] = b
        if "cached_vs_fresh" in name:
            raw[f"{name}_fresh_{step}"] = a
            raw[f"{name}_cached_{step}"] = b
        return row

    for step in range(args.steps):
        source_logits = reference(mx.array([current]), cache=source_cache)
        local_logits = native.model(mx.array([current]), cache=local_cache).logits
        mx.eval(source_logits, local_logits)
        compare("logits", source_logits[:, -1:], local_logits[:, -1:], step)
        expected = int(mx.argmax(source_logits[0, -1]).item())
        actual = int(mx.argmax(local_logits[0, -1]).item())
        tokens.append({"source": expected, "native": actual})
        if step in {0, 15, args.steps - 1}:
            for layer, (a, b) in enumerate(zip(source_cache, local_cache, strict=True)):
                for kind, x, y in zip(["K", "V"], a.state[:2], b.state(), strict=True):
                    compare(f"layer.{layer}.{kind}", x[:, :, : a.offset], y, step)
            fresh = native.model(mx.array([history]), cache=native.model.make_cache()).logits
            mx.eval(fresh)
            compare("native_cached_vs_fresh", fresh[:, -1:], local_logits[:, -1:], step)
            source_fresh = reference(mx.array([history]), cache=make_prompt_cache(reference))
            mx.eval(source_fresh)
            compare("source_cached_vs_fresh", source_fresh[:, -1:], source_logits[:, -1:], step)
        current = [expected]
        history.append(expected)
        if step % 16 == 0:
            print(
                args.model, step, "failures", sum(r["status"] == "FAIL" for r in rows), flush=True
            )
    np.savez_compressed(args.output / "logits.npz", **raw)
    result = {
        "command": sys.argv,
        "model": spec,
        "source_revision": pin["revision"],
        "backend": mx.__version__ if hasattr(mx, "__version__") else "0.32.2",
        "protocol": "source-forced decode, natural EOS observed but forced trace continues",
        "tokens": tokens,
        "comparisons": rows,
        "source_parity": all(
            r["status"] == "PASS"
            for r in rows
            if r["name"] not in {"native_cached_vs_fresh", "source_cached_vs_fresh"}
        ),
        "self_consistency": all(
            r["status"] == "PASS" for r in rows if r["name"] == "native_cached_vs_fresh"
        ),
        "source_self_consistency": all(
            r["status"] == "PASS" for r in rows if r["name"] == "source_cached_vs_fresh"
        ),
        "greedy_agreement": sum(t["source"] == t["native"] for t in tokens),
        "total_steps": args.steps,
        "device": mx.device_info(),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot(
        args.output,
        model=spec,
        status="PASS"
        if result["source_parity"] and result["self_consistency"]
        else "FAIL_NUMERICAL",
    )
    print({k: result[k] for k in ["source_parity", "self_consistency", "greedy_agreement"]})
    return 0 if result["source_parity"] and result["self_consistency"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
