"""Source-versus-primary model-profile distortion, separate from packed-kernel accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from four_path_oracle import load_reference, snapshot
from numerical_metrics import distribution_metrics


def run(args):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(args.checkpoint, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, module = load_reference(bundle, "stored")
        root = Path(__file__).resolve().parents[2]
        if not Path(module.__file__).resolve().is_relative_to(root / "references_libs/mlx-lm-main"):
            raise ValueError("wrong reference")
        source_weight_dtypes = sorted(
            {str(x.dtype) for _, x in tree_flatten(bundle.model.parameters())}
        )
        cases = []
        for prompt in [
            "The capital of France is Paris. The river flows through the city. ",
            "def add(a,b):\n    return a+b\n",
            '{"id":7319,"name":"Ana","valid":true}\n',
        ]:
            seed = bundle.tokenizer.encode(prompt)
            for n in [16, 128, 512]:
                ids = (seed * ((n + 1 + len(seed) - 1) // len(seed)))[: n + 1]
                out = bundle.model(mx.array([ids[:n]])).logits
                value = snapshot(out[:, -1:, :])
                source_logits_dtype = str(out.dtype)
                del out
                cases.append(
                    {
                        "ids": ids,
                        "context": n,
                        "source": value,
                        "source_logits_dtype": source_logits_dtype,
                    }
                )
        bits = int(args.precision[-1])
        for model in [bundle.model, reference]:
            nn.quantize(model, group_size=64, bits=bits, mode="affine")
        bundle = replace(bundle, quantization={"bits": bits, "group_size": 64})
        configure_execution_dtype(bundle, "float16")
        reference.set_dtype(
            mx.float16, predicate=lambda d: d in {mx.float32, mx.float16, mx.bfloat16}
        )
        rows = []
        for c in cases:
            tokens = mx.array([c["ids"][:-1]])
            out = bundle.model(tokens).logits
            ref = reference(tokens)
            mx.eval(out, ref)
            primary = snapshot(out[:, -1:, :])
            ref_last = snapshot(ref[:, -1:, :])
            rows.append(
                {
                    "input_ids": c["ids"][:-1],
                    "context": c["context"],
                    "parts": [c["context"]],
                    "target": c["ids"][-1],
                    "source_logits_dtype": c["source_logits_dtype"],
                    "primary_logits_dtype": str(out.dtype),
                    "reference_logits_dtype": str(ref.dtype),
                    "source_vs_primary": distribution_metrics(
                        c["source"], primary, c["ids"][-1], "float16"
                    ),
                    "same_primary_reference": distribution_metrics(
                        ref_last, primary, c["ids"][-1], "float16"
                    ),
                }
            )
        return {
            "timestamp": time.time(),
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "profile": args.precision,
            "reference_module": module.__file__,
            "reference_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_profile": "stored",
            "source_weight_dtypes": source_weight_dtypes,
            "primary_weight_dtypes": sorted(
                {str(x.dtype) for _, x in tree_flatten(bundle.model.parameters())}
            ),
            "execution": {
                "source": "native natural full forward",
                "primary": "native affine quantized FP16 full forward",
                "reference": "same-primary transformed local reference full forward",
                "kv_policy": "none: all paths uncached",
            },
            "rows": rows,
            "interpretation": (
                "Combined quantization plus floating-activation/parameter policy distortion; "
                "never a framework-only error or a pure weight-only claim"
            ),
            "kernel_accuracy": "Separately measured with exact same packed weights and CPU FP64",
            "status": "MEASURED_NOT_RELEASE_APPROVED",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--precision", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists")
    a.output.write_text(json.dumps(run(a), indent=2) + "\n")


if __name__ == "__main__":
    main()
