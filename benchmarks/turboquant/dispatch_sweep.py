#!/usr/bin/env python3
"""Compare legal dispatch alternatives on identical tensors in one GPU process."""

import argparse
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from operators import measure
from oracle import fixture

from mlx_one.quantization.turboquant.kernels.tq3_expert_qmv import expert_qmv
from mlx_one.quantization.turboquant.kernels.tq3_gather_qmm import tq3_gather_qmm
from mlx_one.quantization.turboquant.kernels.tq3_qmm import tq3_qmm
from mlx_one.quantization.turboquant.kernels.tq3_qmv import tq3_qmv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=20)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "baseline_qmv", a.reference_root / "kernels/polar_qmv.py"
    )
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    result = []
    for o, k in [(512, 2880), (2880, 2880), (4096, 2880), (2880, 4096)]:
        data = {key: mx.array(value) for key, value in fixture(o, k).items()}
        w, s, c = [data[key] for key in ["weight", "scales", "codebook"]]
        for n in [1, 2, 8]:
            x = mx.ones((n, k), mx.float16)
            mx.eval(w, s, c, x)
            fns = {
                "reference_gemv_rows": lambda w=w, s=s, c=c, x=x, n=n: mx.stack(
                    [ref.polar_qmv(w, s, c, x[i], 3, 64) for i in range(n)]
                ),
                "coalesced_gemv_rows": lambda w=w, s=s, c=c, x=x, n=n: mx.stack(
                    [tq3_qmv(w, s, c, x[i], 3, 64) for i in range(n)]
                ),
                "packed_gemm": lambda w=w, s=s, c=c, x=x: tq3_qmm(w, s, c, x, 3, 64),
            }
            for name, fn in fns.items():
                result.append(
                    dict(
                        kind="dense",
                        shape=[n, o, k],
                        strategy=name,
                        **measure(fn, a.repetitions, 2 * n * o * k),
                    )
                )
    o = k = 2880
    e = 32
    data = {key: mx.array(value) for key, value in fixture(o, k, experts=e).items()}
    w, s, c = [data[key] for key in ["weight", "scales", "codebook"]]
    mx.eval(w, s, c)
    for n in [16, 32, 64, 128, 256, 512, 1024]:
        x = mx.ones((n, k), mx.float16)
        idx = mx.array(np.random.default_rng(23).integers(0, e, n, dtype=np.uint32))
        order = mx.argsort(idx)
        mx.argsort(order)
        mx.eval(x, idx)

        # Include sorting/reordering cost in every tiled invocation.
        def tiled(idx=idx, x=x):
            order = mx.argsort(idx)
            inverse = mx.argsort(order)
            return tq3_gather_qmm(w, s, c, x[order], idx[order], 3, 64)[inverse]

        for name, fn in [
            ("coalesced_gather", lambda x=x, idx=idx: expert_qmv(w, s, c, x, idx)),
            ("sorted_tiled_gather", tiled),
        ]:
            result.append(
                dict(
                    kind="expert",
                    shape=[n, o, k, e],
                    strategy=name,
                    **measure(fn, a.repetitions, 2 * n * o * k),
                )
            )
    a.output.write_text(
        json.dumps(dict(schema_version="tq3-dispatch-sweep-v1", results=result), indent=2) + "\n"
    )
    print(a.output)


if __name__ == "__main__":
    main()
