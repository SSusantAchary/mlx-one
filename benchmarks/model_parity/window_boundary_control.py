"""Exact-weight attention-layer boundary controls, independent of full-model oracle.

Phi3's local reference lacks a model-level sliding mask. Its layer is therefore
tested with an explicit source-derived mask. This does not qualify that reference
as an unmodified full-model long-context oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from four_path_oracle import compare, load_reference, snapshot


def reference_retained_before_single_query(cache, offset):
    """Rotating reference returns at most max_size keys including the new query."""
    return min(offset, cache.max_size - 1) if hasattr(cache, "max_size") else offset


def run(path, lengths=None):
    import mlx.core as mx
    from local_reference import reference_module

    from mlx_one.models.shared.decoder import attention_mask_for
    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, module = load_reference(bundle, "stored")
        config = bundle.model.config
        window = config.sliding_window
        nc_factory = bundle.model.make_cache

        def rc_factory():
            return reference_module("cache").make_prompt_cache(reference)

        layers = [0, 5] if bundle.architecture == "gemma3_text" else [0]
        rows = []
        for layer_index in layers:
            layer = bundle.model.model.layers[layer_index]
            ref_layer = reference.model.layers[layer_index]
            actual_window = window if layer_index == 0 else None
            for length in lengths or (window - 1, window, window + 1):
                ids = mx.array([bundle.tokenizer.encode("The capital of France is") * length])
                hidden = bundle.model.model.embed_tokens(ids[:, :length])
                if bundle.architecture == "gemma3_text":
                    # The attention-layer input is common to both implementations.
                    hidden = hidden * mx.array(config.hidden_size**0.5, mx.bfloat16)
                hidden = layer.input_layernorm(hidden)
                dtype = str(hidden.dtype).rsplit(".", 1)[-1]
                full_mask = attention_mask_for(
                    length, 0, 0, window=actual_window, dtype=hidden.dtype
                )

                def native(values, cache=None, mask=None, layer=layer):
                    if bundle.architecture == "phi3":
                        return layer.self_attn(values, cache=cache, mask=mask)
                    return layer.self_attn(values, cache=cache)

                full = native(hidden, mask=full_mask)
                ref_full = ref_layer.self_attn(hidden, mask=full_mask)
                mx.eval(full, ref_full)
                nc, rc = nc_factory()[layer_index], rc_factory()[layer_index]
                prefix_mask = attention_mask_for(
                    length - 1, 0, 0, window=actual_window, dtype=hidden.dtype
                )
                mx.eval(native(hidden[:, :-1], nc, prefix_mask))
                mx.eval(ref_layer.self_attn(hidden[:, :-1], mask=prefix_mask, cache=rc))
                retained = min(length - 1, getattr(nc, "max_size", length - 1))
                last_mask = attention_mask_for(
                    1, length - 1, retained, window=actual_window, dtype=hidden.dtype
                )
                final = native(hidden[:, -1:], nc, last_mask)
                # Rotating reference evicts before returning a single-query view.
                ref_retained = reference_retained_before_single_query(rc, length - 1)
                ref_last_mask = attention_mask_for(
                    1, length - 1, ref_retained, window=actual_window, dtype=hidden.dtype
                )
                ref_final = ref_layer.self_attn(hidden[:, -1:], mask=ref_last_mask, cache=rc)
                mx.eval(final, ref_final)
                row = {
                    "layer": layer_index,
                    "window": actual_window,
                    "tokens": length,
                    "absolute_position": length - 1,
                    "native_offset": nc.offset,
                    "reference_offset": rc.offset,
                    "native_retained_keys": int(nc.state()[0].shape[2]),
                    "cross_full": compare(snapshot(ref_full), snapshot(full), dtype),
                    "cross_cached": compare(snapshot(ref_final), snapshot(final), dtype),
                    "native_full_vs_cached_last": compare(
                        snapshot(full[:, -1:]), snapshot(final), dtype
                    ),
                    "reference_full_vs_cached_last": compare(
                        snapshot(ref_full[:, -1:]), snapshot(ref_final), dtype
                    ),
                }
                assert nc.offset == rc.offset == length
                assert nc.state()[0].shape[2] == min(length, actual_window or length)
                rows.append(row)
        return {
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "reference_source": {
                "file": module.__file__,
                "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            },
            "rows": rows,
            "scope": "Exact-weight attention layers, common inputs, explicit source window masks",
            "full_model_long_context_qualified": False,
            "status": "PASS"
            if all(
                r[key]["status"] == "PASS"
                for r in rows
                for key in (
                    "cross_full",
                    "cross_cached",
                    "native_full_vs_cached_last",
                    "reference_full_vs_cached_last",
                )
            )
            else "FAIL_NUMERICAL",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", help="Optional comma-separated boundary lengths")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = run(
        args.checkpoint, [int(n) for n in args.lengths.split(",")] if args.lengths else None
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["checkpoint"], result["status"], flush=True)


if __name__ == "__main__":
    main()
