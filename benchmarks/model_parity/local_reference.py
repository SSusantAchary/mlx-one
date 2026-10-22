"""Offline, B1 comparison against the required local MLX-LM snapshot.

This development harness does not import the installed mlx_lm package or change
reference files. It loads only the snapshot's model submodules under an isolated
namespace, avoiding its missing generated version and unrelated CLI imports.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import resource
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def reference_module(family: str):
    namespace = "_mlx_one_revamp_local_reference"
    snapshot = ROOT / "references_libs/mlx-lm-main/mlx_lm"
    if not snapshot.is_dir():
        raise FileNotFoundError(f"required local reference is missing: {snapshot}")
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(snapshot)]
        sys.modules[namespace] = package
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(f"{namespace}.models.{family}")
    finally:
        sys.dont_write_bytecode = previous


def compare(reference, candidate, gate_dtype):
    import mlx.core as mx
    import numpy as np

    from mlx_one.ops.numerical_trace import compare_tensors

    mx.eval(reference, candidate)
    return compare_tensors(np.asarray(reference.astype(mx.float32)),
                           np.asarray(candidate.astype(mx.float32)), dtype=gate_dtype)


def run(checkpoint, *, dtype, prompt_tokens, generated_tokens):
    import mlx.core as mx

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.reporting import tensor_dtypes
    from mlx_one.runtime_baseline import file_digest
    from mlx_one.text.loading import (
        _prepare_quantized_model,
        _read_safetensors,
        _validate_quantized_tensors,
        load_text_model,
    )

    started = time.perf_counter()
    bundle = load_text_model(checkpoint, offline=True)
    load_seconds = time.perf_counter() - started
    configure_execution_dtype(bundle, dtype)
    family = bundle.architecture
    if family not in {"gpt2", "qwen2", "qwen3", "lfm2", "qwen3_5", "llama"}:
        raise ValueError(f"local reference adapter is unavailable for {family}")
    data = json.loads((bundle.path / "config.json").read_text())
    module = reference_module(family)
    reference = module.Model(module.ModelArgs.from_dict(data))
    tensors = reference.sanitize(_read_safetensors(bundle.path))
    if bundle.quantization:
        _prepare_quantized_model(reference, tensors, bundle.quantization)
        _validate_quantized_tensors(reference, tensors)
    reference.load_weights(list(tensors.items()), strict=True)
    if dtype:
        reference.set_dtype(getattr(mx, dtype), predicate=lambda d:
                            str(d).rsplit(".", 1)[-1] in {"float16", "bfloat16", "float32"})
    reference.eval()
    mx.eval(reference.parameters(), bundle.model.parameters())
    cache_module = reference_module("cache")
    seed = bundle.tokenizer.encode("The capital of France is")
    prompt = (seed * ((prompt_tokens + len(seed) - 1) // len(seed)))[:prompt_tokens]
    if prompt_tokens + generated_tokens > bundle.context_length:
        raise ValueError("requested context/output exceeds checkpoint positional limit")
    inputs = mx.array([prompt])
    native_cache = bundle.model.make_cache()
    reference_cache = cache_module.make_prompt_cache(reference)
    native_full = bundle.model(inputs).logits[:, -1:, :]
    reference_full = reference(inputs)[:, -1:, :]
    gate = dtype or str(native_full.dtype).rsplit(".", 1)[-1]
    boundaries = [{"name": "fresh_prefill", **compare(reference_full, native_full, gate)}]
    native_output = bundle.model(inputs, cache=native_cache).logits[:, -1:, :]
    reference_output = reference(inputs, cache=reference_cache)[:, -1:, :]
    boundaries.append({"name": "cached_prefill", **compare(reference_output, native_output, gate)})
    continuation = {"native": compare(native_full, native_output, gate),
                    "reference": compare(reference_full, reference_output, gate)}
    native_ids, reference_ids = [], []
    # Follow the reference token in both models: differences cannot change inputs.
    for step in range(generated_tokens):
        native_ids.append(int(mx.argmax(native_output[0, -1]).item()))
        reference_ids.append(int(mx.argmax(reference_output[0, -1]).item()))
        token = mx.array([[reference_ids[-1]]])
        native_output = bundle.model(token, cache=native_cache).logits[:, -1:, :]
        reference_output = reference(token, cache=reference_cache)[:, -1:, :]
        boundaries.append({"name": f"forced_decode_{step}",
                           **compare(reference_output, native_output, gate)})
    # A full forward over the same continuation checks the final cache boundary.
    continued = mx.array([prompt + reference_ids])
    continuation["native_final"] = compare(
        bundle.model(continued).logits[:, -1:, :], native_output, gate)
    continuation["reference_final"] = compare(
        reference(continued)[:, -1:, :], reference_output, gate)
    reference_path = Path(module.__file__)
    report = {
        "schema_version": "revamp-local-reference-v1", "status": "PASS" if all(
            x["status"] == "PASS" for x in [*boundaries, *continuation.values()]
        ) and native_ids == reference_ids else "FAIL",
        "scope": {"model_id": bundle.model_id, "revision": bundle.revision,
                  "architecture": family, "mode": "reference_dense", "batch_size": 1,
                  "prompt_tokens": prompt_tokens, "forced_decode_steps": generated_tokens,
                  "execution_dtype": dtype, "quantization": bundle.quantization,
                  "actual_weight_dtypes": tensor_dtypes(bundle.model.parameters()),
                  "actual_logits_dtype": str(native_full.dtype)},
        "reference": {"path": str(reference_path.relative_to(ROOT)),
                      "sha256": file_digest(reference_path),
                      "installed_mlx_lm_used": False},
        "checkpoint_files": {p.name: file_digest(p) for p in sorted(bundle.path.glob("*"))
                             if p.is_file() and p.suffix in {".safetensors", ".json"}},
        "boundaries": boundaries, "full_vs_cached": continuation,
        "native_greedy_ids_on_reference_continuation": native_ids,
        "reference_greedy_ids": reference_ids,
        "native_cache_offsets": [c.offset for c in native_cache],
        "cold_native_load_seconds": load_seconds,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "duration_seconds": time.perf_counter() - started,
        "qualification": "DIAGNOSTIC_ONLY", "performance_qualified": False,
        "notes": "Forced token comparison; EOS, free generation and HTTP are separate gates.",
    }
    del bundle, reference, tensors, native_cache, reference_cache
    gc.collect()
    mx.clear_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"])
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new evidence path")
    if min(args.prompt_tokens, args.generated_tokens) < 1:
        parser.error("prompt and output sizes must be positive")
    try:
        report = run(args.checkpoint, dtype=args.dtype, prompt_tokens=args.prompt_tokens,
                     generated_tokens=args.generated_tokens)
    except Exception as exc:
        report = {"schema_version": "revamp-local-reference-v1", "status": "BLOCKED",
                  "reason": f"{type(exc).__name__}: {exc}", "qualification": "NOT_QUALIFIED"}
    report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src/mlx_one").rglob("*.py"))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "qualification")
                      if k in report} | {"output": str(args.output)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
