"""Trace packed gathered projections that bypass module __call__ hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path


@contextmanager
def capture_with_projections(model, family):
    import mlx.core as mx
    from layerwise import canonical_name, capture_modules

    from mlx_one.models.shared import execution

    selected = {id(module): canonical_name(name, family) for name, module in model.named_modules()}
    original = execution._project
    with capture_modules(model, family) as events:

        def append(name, value):
            snapshot = mx.zeros_like(value)
            snapshot[...] = value
            mx.eval(snapshot)
            occurrence = 1 + sum(key.startswith(name + "#") for key, _ in events)
            events.append((f"{name}#{occurrence}", snapshot))

        def project(module, value, *, execution_dtype=None, as_linear=False):
            bypasses = (
                execution_dtype == "float16"
                and getattr(module, "bits", None) in {4, 8}
                and getattr(module, "mode", None) == "affine"
                and not as_linear
            )
            name = selected.get(id(module))
            if bypasses and name and name.rsplit(".", 1)[-1] in {"o_proj", "c_proj", "out_proj"}:
                append(name + ".input", value)
            result = original(module, value, execution_dtype=execution_dtype, as_linear=as_linear)
            if bypasses and name:
                append(name, result)
            return result

        execution._project = project
        try:
            yield events
        finally:
            execution._project = original


def run(path, precision):
    import mlx.core as mx
    import mlx.nn as nn
    from layerwise import compare_events
    from local_reference import reference_module

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.text.loading import _read_safetensors

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(1024**3)
    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        family = bundle.architecture
        source = reference_module(family)
        config = json.loads((bundle.path / "config.json").read_text())
        reference = source.Model(source.ModelArgs.from_dict(config))
        weights = _read_safetensors(bundle.path)
        if hasattr(reference, "sanitize"):
            weights = reference.sanitize(weights)
        reference.load_weights(list(weights.items()), strict=True)
        del weights
        bits = int(precision[-1])
        for model in (bundle.model, reference):
            nn.quantize(model, group_size=64, bits=bits, mode="affine")
            model.eval()
        bundle = replace(bundle, quantization={"bits": bits, "group_size": 64})
        configure_execution_dtype(bundle, "float16")
        reference.set_dtype(
            mx.float16,
            predicate=lambda dtype: (
                str(dtype).rsplit(".", 1)[-1] in {"float16", "bfloat16", "float32"}
            ),
        )
        mx.eval(reference.parameters(), bundle.model.parameters())
        seed = bundle.tokenizer.encode("The capital of France is")
        prompt = (seed * ((16 + len(seed) - 1) // len(seed)))[:16]
        tokens = mx.array([prompt])
        with capture_with_projections(reference, family) as expected:
            ref_logits = reference(tokens)
        with capture_with_projections(bundle.model, family) as actual:
            native_logits = bundle.model(tokens).logits
        mx.eval(ref_logits, native_logits)
        return {
            "model_id": bundle.model_id,
            "revision": bundle.revision,
            "precision": precision,
            "execution_dtype": "float16",
            "trace": compare_events(expected, actual, "float16"),
            "scope": (
                "Identical exact-source tensors, in-memory affine transform, "
                "full 16-token forward only"
            ),
            "reference_source_sha256": hashlib.sha256(
                Path(source.__file__).read_bytes()
            ).hexdigest(),
            "qualification": "NOT_QUALIFIED",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--precision", choices=("affine_q4", "affine_q8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = run(args.checkpoint, args.precision)
    root = Path(__file__).resolve().parents[2]
    report["source_hashes"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for base in ("src/mlx_one", "benchmarks/model_parity")
        for p in (root / base).rglob("*.py")
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
