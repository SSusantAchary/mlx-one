"""Remove only universally invisible KV columns in a diagnostic Gemma invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from four_path_oracle import load_reference, snapshot
from numerical_metrics import distribution_metrics


def run(args):
    import mlx.core as mx
    from local_reference import reference_module

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(args.checkpoint, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, module = load_reference(bundle, "stored")
        root = Path(__file__).resolve().parents[2]
        if not Path(module.__file__).resolve().is_relative_to(root / "references_libs/mlx-lm-main"):
            raise ValueError("wrong reference")
        result = {
            "timestamp": time.time(),
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "profile": "stored",
            "reference_module": module.__file__,
            "reference_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rows": [],
            "production_modified": False,
        }
        seed = bundle.tokenizer.encode("The red book was on the table. The city is calm. ")

        class View:
            def __init__(self, owner):
                self.owner = owner

            def __getattr__(self, key):
                return getattr(self.owner, key)

            def update_and_fetch(self, k, v):
                d = k.dtype
                k, v = self.owner.update_and_fetch(k.astype(mx.float16), v.astype(mx.float16))
                return k.astype(d), v.astype(d)

        cases = [
            (n, [n // 3, n // 3, n - 2 * (n // 3)], "thirds")
            for n in [16, 128, 511, 512, 513, 1024, 2048]
        ]
        cases.extend((n, [128] * (n // 128), "blocks128") for n in [1024, 2048])
        for n, parts, partition_label in cases:
            ids = (seed * ((n + 1 + len(seed) - 1) // len(seed)))[: n + 1]
            outputs = {}
            controls = []
            for mode in ["canonical", "drop_invisible", "drop_invisible_contiguous", "reference"]:
                cache = (
                    configure_kv_caches(bundle.model.make_cache(), "f16", "f16")
                    if mode != "reference"
                    else [View(c) for c in reference_module("cache").make_prompt_cache(reference)]
                )
                begin = 0
                for width in parts:
                    original = mx.fast.scaled_dot_product_attention
                    index = 0

                    def sdpa(
                        q,
                        k,
                        v,
                        *,
                        scale,
                        mask=None,
                        original=original,
                        mode=mode,
                        controls=controls,
                        **kw,
                    ):
                        nonlocal index
                        layer = index
                        index += 1
                        if (
                            not mode.startswith("drop_invisible")
                            or mask is None
                            or isinstance(mask, str)
                        ):
                            return original(q, k, v, scale=scale, mask=mask, **kw)
                        raw = snapshot(mask)
                        visibility = raw if raw.dtype == bool else np.isfinite(raw)
                        # Only discard columns invisible to every query, never visible state.
                        axes = tuple(range(visibility.ndim - 1))
                        allowed = visibility.any(axis=axes)
                        remove = 0
                        for visible in allowed:
                            if visible:
                                break
                            remove += 1
                        if not remove:
                            return original(q, k, v, scale=scale, mask=mask, **kw)
                        baseline = original(q, k, v, scale=scale, mask=mask, **kw)
                        reduced_k, reduced_v = k[:, :, remove:], v[:, :, remove:]
                        reduced_mask = mask[..., remove:]
                        if mode == "drop_invisible_contiguous":
                            reduced_k, reduced_v, reduced_mask = (
                                mx.contiguous(x) for x in (reduced_k, reduced_v, reduced_mask)
                            )
                        reduced = original(
                            q,
                            reduced_k,
                            reduced_v,
                            scale=scale,
                            mask=reduced_mask,
                            **kw,
                        )
                        checks = compare_tensors(
                            snapshot(baseline[:, :, -1:]),
                            snapshot(reduced[:, :, -1:]),
                            dtype=str(q.dtype).rsplit(".", 1)[-1],
                        )
                        controls.append(
                            {
                                "layer_attention_index": layer,
                                "original_key_length": k.shape[2],
                                "removed_fully_masked_columns": remove,
                                "query_length": q.shape[2],
                                "mode": mode,
                                "removed_probability_mass_exactly_zero_by_mask": True,
                                "output_difference": checks,
                            }
                        )
                        return reduced

                    if mode != "reference":
                        mx.fast.scaled_dot_product_attention = sdpa
                    try:
                        token = mx.array([ids[begin : begin + width]])
                        out = (
                            reference(token, cache=cache)
                            if mode == "reference"
                            else bundle.model(token, cache=cache).logits
                        )
                        mx.eval(out)
                        last = snapshot(out[:, -1:])
                        del out
                    finally:
                        mx.fast.scaled_dot_product_attention = original
                    begin += width
                outputs[mode] = last
                del cache
                mx.clear_cache()
            result["rows"].append(
                {
                    "context": n,
                    "parts": parts,
                    "partition_label": partition_label,
                    "input_ids": ids[:-1],
                    "reference_vs_canonical": distribution_metrics(
                        outputs["reference"], outputs["canonical"], ids[-1], "bfloat16"
                    ),
                    "reference_vs_drop_invisible": distribution_metrics(
                        outputs["reference"], outputs["drop_invisible"], ids[-1], "bfloat16"
                    ),
                    "canonical_vs_drop_invisible": distribution_metrics(
                        outputs["canonical"], outputs["drop_invisible"], ids[-1], "bfloat16"
                    ),
                    "reference_vs_drop_contiguous": distribution_metrics(
                        outputs["reference"],
                        outputs["drop_invisible_contiguous"],
                        ids[-1],
                        "bfloat16",
                    ),
                    "drop_view_vs_drop_contiguous": distribution_metrics(
                        outputs["drop_invisible"],
                        outputs["drop_invisible_contiguous"],
                        ids[-1],
                        "bfloat16",
                    ),
                    "controls": controls,
                }
            )
        result["status"] = "MEASURED_NOT_RELEASE_APPROVED"
        return result

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists")
    a.output.write_text(json.dumps(run(a), indent=2) + "\n")


if __name__ == "__main__":
    main()
