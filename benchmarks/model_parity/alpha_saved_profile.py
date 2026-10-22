"""Persist the exact tested transform and prove packed/tensor round-trip identity."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path


def run(source, destination, precision):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mlx_one.conversion import _save_shards, write_conversion_manifest
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.text.loading import load_text_model

    if destination.exists():
        raise FileExistsError(destination)
    mx.set_memory_limit(20 * 1024**3)
    bundle = load_text_model(source, offline=True)
    bits = int(precision[-1])
    nn.quantize(bundle.model, group_size=64, bits=bits, mode="affine")
    from dataclasses import replace

    bundle = replace(bundle, quantization={"bits": bits, "group_size": 64, "mode": "affine"})
    configure_execution_dtype(bundle, "float16")
    parameters = dict(tree_flatten(bundle.model.parameters()))
    mx.eval(parameters)

    def hashes(parameters):
        import numpy as np

        return {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(np.array(value).tobytes()).hexdigest(),
            }
            for name, value in parameters.items()
        }

    before = hashes(parameters)
    destination.mkdir(parents=True)
    for path in bundle.path.iterdir():
        if path.is_file() and path.suffix in {".json", ".jinja", ".txt", ".model", ".md"}:
            shutil.copy2(path, destination / path.name)
    config = json.loads((destination / "config.json").read_text())
    config["quantization"] = {
        "bits": bits,
        "group_size": 64,
        "mode": "affine",
        "quant_method": "mlx",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    _save_shards(mx, destination, parameters, 5 * 1024**3)
    manifest = write_conversion_manifest(
        destination,
        {
            "schema_version": 2,
            "source": bundle.model_id,
            "revision": bundle.revision,
            "precision": f"int{bits}-affine",
            "group_size": 64,
            "converter": "alpha-exact-transform-save-v1",
            "floating_dtype": "float16",
            "transform_order": "quantize stored values, then cast floating tensors to FP16",
        },
    )
    del bundle, parameters
    gc.collect()
    mx.clear_cache()
    loaded = load_text_model(destination, offline=True)
    after = hashes(dict(tree_flatten(loaded.model.parameters())))
    return {
        "timestamp": time.time(),
        "status": "PASS" if before == after else "FAIL",
        "destination": str(destination),
        "checkpoint": loaded.model_id,
        "revision": loaded.revision,
        "profile": precision,
        "manifest": manifest,
        "source_parameter_hashes": before,
        "reloaded_parameter_hashes": after,
        "tensor_and_packed_round_trip_exact": before == after,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    try:
        result = run(args.checkpoint, args.destination, args.precision)
    except Exception as exc:
        import traceback

        result = {"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
