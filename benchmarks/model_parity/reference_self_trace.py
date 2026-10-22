"""Trace each model's first recorded full/cache discrepancy with correct token axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from four_path_oracle import load_reference, prefix_trace


def run(path, oracle):
    import mlx.core as mx
    from local_reference import reference_module

    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, _ = load_reference(bundle, oracle["precision"])
        tokens = mx.array([oracle["token_ids"]])
        rows = {}
        for label, model, pair, native in (
            ("native", bundle.model, "A_C", True),
            ("reference", reference, "B_D", False),
        ):
            first = oracle["pairwise"][pair]["first_nonzero_position"] or len(oracle["token_ids"])
            factory = (
                model.make_cache
                if native
                else lambda model=model: reference_module("cache").make_prompt_cache(model)
            )
            rows[label] = {
                "prefix_length": first,
                "trace": prefix_trace(
                    model,
                    bundle.architecture,
                    native,
                    tokens,
                    first,
                    factory,
                    oracle["actual_logits_dtype"],
                ),
            }
        return {
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "precision": oracle["precision"],
            "token_ids": oracle["token_ids"],
            "trace_contract": "Materialized module boundaries; semantic sequence axis, "
            "not an assumed fixed axis for every four-dimensional norm.",
            "rows": rows,
            "qualification": "DIAGNOSTIC_ONLY",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = run(args.checkpoint, json.loads(args.oracle.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        result["checkpoint"],
        {k: v["trace"]["first_nonzero"] for k, v in result["rows"].items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
