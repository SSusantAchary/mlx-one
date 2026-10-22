"""Independent FP64 operation and analytical cache controls on real MLX."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from types import SimpleNamespace

import numpy as np


def array(value):
    import mlx.core as mx

    mx.eval(value)
    return np.array(value.astype(mx.float32)).astype(np.float64)


def errors(expected, actual):
    delta = np.abs(expected - actual)
    meaningful = np.abs(expected) > max(1e-3, np.max(np.abs(expected)) * 1e-3)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "relative_floor": max(1e-3, float(np.max(np.abs(expected))) * 1e-3),
        "max_relative_above_floor": float((delta[meaningful] / np.abs(expected[meaningful])).max())
        if meaningful.any()
        else None,
    }


def operation_controls():
    import mlx.core as mx
    import mlx.nn as nn

    rng = np.random.default_rng(1947)
    rows = []
    for dtype, unit in (("float32", 2**-24), ("float16", 2**-11), ("bfloat16", 2**-8)):
        for difficult in (False, True):
            x = rng.normal(size=(16, 64)) / 8
            w = rng.normal(size=(96, 64)) / 8
            if difficult:
                x[:, ::2] *= -100
                w[:, ::2] *= 0.01
            a, b = (mx.array(z, dtype=getattr(mx, dtype)) for z in (x, w))
            a64, b64 = array(a), array(b)
            for bits in (None, 4, 8):
                b64 = array(b)
                expanded = np.abs(b64)
                if bits:
                    packed, scales, biases = mx.quantize(b, group_size=64, bits=bits)
                    words = np.array(packed).astype(np.uint32)
                    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
                    codes = ((words[..., None] >> shifts) & ((1 << bits) - 1)).reshape(96, 64)
                    scale64, bias64 = array(scales), array(biases)
                    b64 = codes * scale64 + bias64
                    expanded = codes * np.abs(scale64) + np.abs(bias64)

                    def multiply(v, packed=packed, scales=scales, biases=biases, bits=bits):
                        return mx.quantized_matmul(
                            v, packed, scales=scales, biases=biases, group_size=64, bits=bits
                        )
                else:

                    def multiply(v, b=b):
                        return v @ b.T

                expected = a64 @ b64.T
                # Standard dot-product roundoff bound plus final output rounding.
                gamma = (128 * 2**-24) / (1 - 128 * 2**-24)
                bound = gamma * (np.abs(a64) @ np.abs(b64).T) + 2 * unit * np.abs(expected)
                affine_bound = (gamma + 3 * unit / (1 - 3 * unit)) * (
                    np.abs(a64) @ expanded.T
                ) + 2 * unit * np.abs(expected)
                full_output = multiply(a)
                actual_output_dtype = str(full_output.dtype).rsplit(".", 1)[-1]
                full = array(full_output)
                single = np.concatenate([array(multiply(a[i : i + 1])) for i in range(16)])
                for shape, actual in (("gemm", full), ("gemv", single)):
                    rows.append(
                        {
                            "id": f"dot-{dtype}-{difficult}-{bits}-{shape}",
                            "dtype": dtype,
                            "operation": "packed_matmul" if bits else "dense_matmul",
                            "input_dtype": str(a.dtype).rsplit(".", 1)[-1],
                            "weight_dtype": str(packed.dtype if bits else b.dtype).rsplit(".", 1)[
                                -1
                            ],
                            "accumulator_dtype": None,
                            "accumulator_evidence": (
                                "Not observable from public dispatch; assumption tested separately"
                            ),
                            "output_dtype": actual_output_dtype,
                            "quantized": bool(bits),
                            "bits": bits,
                            "independent_weight_equation": "codes*scale+bias in NumPy FP64"
                            if bits
                            else "stored floating values",
                            "bound_interpretation": (
                                "Diagnostic hypothesis, not an established packed-kernel contract"
                            )
                            if bits
                            else "FP32 accumulation hypothesis plus output rounding",
                            "status": "PASS"
                            if np.all(np.abs(actual - expected) <= bound + 1e-6)
                            else "FAIL",
                            **errors(expected, actual),
                            "max_theoretical_bound": float(bound.max()),
                            "alternative_affine_low_precision_bound": float(affine_bound.max()),
                            "alternative_bound_pass": bool(
                                np.all(np.abs(actual - expected) <= affine_bound + 1e-6)
                            ),
                            "alternative_bound_scope": "Assumes three low-precision affine "
                            "roundings and FP32 dot accumulation; diagnostic, not a proven "
                            "statement of the Metal kernel or release acceptance.",
                            "full_decode_shape_max": float(np.max(np.abs(full - single))),
                        }
                    )
            norm = nn.RMSNorm(64, eps=1e-6)
            norm.weight = mx.ones((64,), dtype=getattr(mx, dtype))
            expected = a64 / np.sqrt(np.mean(a64**2, axis=-1, keepdims=True) + 1e-6)
            norm_output = norm(a)
            actual = array(norm_output)
            bound = (4 * unit + 128 * 2**-24) * max(1, np.abs(expected).max())
            rows.append(
                {
                    "id": f"rms-{dtype}-{difficult}",
                    "operation": "rms_norm",
                    "input_dtype": str(a.dtype).rsplit(".", 1)[-1],
                    "weight_dtype": str(norm.weight.dtype).rsplit(".", 1)[-1],
                    "output_dtype": str(norm_output.dtype).rsplit(".", 1)[-1],
                    "accumulator_dtype": None,
                    "status": "PASS" if np.max(np.abs(expected - actual)) <= bound else "FAIL",
                    **errors(expected, actual),
                    "max_theoretical_bound": float(bound),
                }
            )
            q, k, v = [
                mx.array(rng.normal(size=(1, h, 8, 64)) / 4, dtype=getattr(mx, dtype))
                for h in (4, 2, 2)
            ]
            q64, k64, v64 = array(q), np.repeat(array(k), 2, axis=1), np.repeat(array(v), 2, axis=1)
            visibility = np.arange(8)[None, :] <= np.arange(8)[:, None]
            scores = q64 @ k64.swapaxes(-1, -2) / 8
            scores = np.where(visibility, scores, -np.inf)
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            expected = probability @ v64
            bound = (8 * unit + 256 * 2**-24) * max(1, np.abs(v64).max())
            fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask="causal")
            explicit = mx.softmax(
                mx.where(
                    mx.array(visibility),
                    (
                        q.astype(mx.float32)
                        @ mx.repeat(k, 2, axis=1).astype(mx.float32).swapaxes(-1, -2)
                    )
                    / 8,
                    -float("inf"),
                ),
                axis=-1,
            ) @ mx.repeat(v, 2, axis=1).astype(mx.float32)
            for mode, output in (("fused", fused), ("unfused-fp32", explicit)):
                actual = array(output)
                rows.append(
                    {
                        "id": f"attention-{dtype}-{difficult}-{mode}",
                        "operation": "attention",
                        "input_dtype": str(q.dtype).rsplit(".", 1)[-1],
                        "weight_dtype": None,
                        "output_dtype": str(output.dtype).rsplit(".", 1)[-1],
                        "accumulator_dtype": None,
                        "status": "PASS" if np.max(np.abs(actual - expected)) <= bound else "FAIL",
                        **errors(expected, actual),
                        "max_theoretical_bound": float(bound),
                    }
                )
    return rows


def cache_controls(family, *, local=False):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.core.cache import KVCache
    from mlx_one.core.sliding_cache import SlidingKVCache
    from mlx_one.models.shared.decoder import attention_mask_for

    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        n_embd=64,
        n_head=4,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        norm_eps=1e-6,
        attention_bias=False,
        rope_traditional=False,
        layer_types=["sliding_attention" if local else "full_attention"],
        rope_local_base_freq=10000.0,
        sliding_window=7,
        query_pre_attn_scalar=16.0,
        attn_logit_softcapping=None,
        rotary_dim=16,
    )
    if family in {"gemma3", "phi3"}:
        module = importlib.import_module(f"mlx_one.models.language.{family}.model")
        block = (
            module.Gemma3Attention(config, 0)
            if family == "gemma3"
            else module.Phi3Attention(config)
        )
    else:
        module = importlib.import_module(f"mlx_one.models.language.{family}.reference_math")
        block = module.Attention(config)
    for _, layer in block.named_modules():
        if isinstance(layer, nn.Linear):
            shape = layer.weight.shape
            layer.weight = mx.array(
                ((np.arange(np.prod(shape)).reshape(shape) % 9) - 4) / 128, dtype=mx.float32
            )
            if "bias" in layer:
                layer.bias = mx.zeros(layer.bias.shape)
    source = ((np.arange(23 * 64).reshape(1, 23, 64) % 17) - 8) / 32
    owner = SlidingKVCache(7) if local else KVCache()
    records, expected_k, expected_v, captured = [], [], [], []
    original = mx.fast.scaled_dot_product_attention

    def capture(q, k, v, *, scale, mask=None, **kwargs):
        captured.append((array(q), array(k), array(v), mask))
        return original(q, k, v, scale=scale, mask=mask, **kwargs)

    def projected(x, layer):
        result = x @ array(layer.weight).T
        if "bias" in layer:
            result += array(layer.bias)
        return result

    def rotary_np(x, positions):
        angles = positions[:, None] / 10000.0 ** (np.arange(0, 16, 2) / 16)
        first, second = np.split(x, 2, axis=-1)
        return np.concatenate(
            (
                first * np.cos(angles) - second * np.sin(angles),
                second * np.cos(angles) + first * np.sin(angles),
            ),
            axis=-1,
        )

    mx.fast.scaled_dot_product_attention = capture
    try:
        offset = 0
        for length in (3, 1, 8, 2, 9):
            hidden = source[:, offset : offset + length]
            if family == "gpt2":
                q, k, v = np.split(projected(hidden, block.c_attn), 3, axis=-1)
                heads = 4
            elif family == "phi3":
                q, k, v = np.split(projected(hidden, block.qkv_proj), [64, 96], axis=-1)
                heads = 2
            else:
                q, k, v = [
                    projected(hidden, getattr(block, n)) for n in ("q_proj", "k_proj", "v_proj")
                ]
                heads = 2
            q = q.reshape(1, length, 4, 16).transpose(0, 2, 1, 3)
            k, v = [x.reshape(1, length, heads, 16).transpose(0, 2, 1, 3) for x in (k, v)]
            if family in {"qwen3", "gemma3", "lfm2"}:
                q = q / np.sqrt(np.mean(q**2, axis=-1, keepdims=True) + 1e-6)
                k = k / np.sqrt(np.mean(k**2, axis=-1, keepdims=True) + 1e-6)
            if family != "gpt2":
                q, k = [rotary_np(x, np.arange(offset, offset + length)) for x in (q, k)]
            expected_k.append(k)
            expected_v.append(v)
            retained = min(offset, 7) if local else offset
            mask = attention_mask_for(length, offset, retained, window=7 if local else None)
            settings = {"cache": owner}
            if family in {"gemma3"}:
                pass
            elif family == "phi3":
                settings["mask"] = mask
            else:
                settings["mask"] = mask
            result = block(mx.array(hidden, dtype=mx.float32), **settings)
            mx.eval(result)
            cq, ck, cv, actual_mask = captured[-1]
            allk, allv = [np.concatenate(x, axis=2) for x in (expected_k, expected_v)]
            start = offset - retained
            metrics = [
                errors(q, cq),
                errors(allk[:, :, start:], ck),
                errors(allv[:, :, start:], cv),
            ]
            query = np.arange(offset, offset + length)[:, None]
            key = np.arange(start, offset + length)[None, :]
            visible = (key <= query) & ((key > query - 7) if local else True)
            if isinstance(actual_mask, str):
                observed = np.arange(ck.shape[2])[None, :] <= np.arange(length)[:, None] + retained
            elif actual_mask is None:
                observed = np.ones(visible.shape, bool)
            else:
                mask_array = np.array(actual_mask)
                observed = mask_array if mask_array.dtype == bool else np.isfinite(mask_array)
            stored_k, stored_v = owner.state()
            # SlidingKVCache.state uses absolute offset but slices safely to retained state.
            tail = min(offset + length, 7) if local else offset + length
            state_error = max(
                np.max(np.abs(array(stored_k) - allk[:, :, -tail:])),
                np.max(np.abs(array(stored_v) - allv[:, :, -tail:])),
            )
            records.append(
                {
                    "offset_before": offset,
                    "input_length": length,
                    "query_positions": query[:, 0].tolist(),
                    "key_positions": key[0].tolist(),
                    "rotary_offset": offset,
                    "offset_after": owner.offset,
                    "qkv_errors": metrics,
                    "state_error": float(state_error),
                    "visibility_exact": bool(np.array_equal(visible, observed)),
                    "status": "PASS"
                    if max(m["max_abs"] for m in metrics) < 1e-4
                    and state_error < 1e-4
                    and np.array_equal(visible, observed)
                    and owner.offset == offset + length
                    else "FAIL",
                }
            )
            offset += length
        owner.reset()
        assert owner.offset == 0 and owner.state() == (None, None)
        fresh = KVCache()
        assert fresh.offset == 0 and fresh.state() == (None, None)
    finally:
        mx.fast.scaled_dot_product_attention = original
    return {
        "family": family,
        "local": local,
        "analytical_weight_identity": "pattern/128",
        "configuration": vars(config),
        "partitions": [3, 1, 8, 2, 9],
        "reset": "PASS",
        "records": records,
        "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path

    path = Path(args.output)
    if path.exists():
        parser.error("output exists")
    operations = operation_controls()
    caches = [cache_controls(f) for f in ("gpt2", "qwen2", "qwen3", "llama", "lfm2", "phi3")]
    caches += [cache_controls("gemma3", local=local) for local in (False, True)]
    result = {
        "timestamp": time.time(),
        "operations": operations,
        "caches": caches,
        "status": "PASS" if all(r["status"] == "PASS" for r in operations + caches) else "FAIL",
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], len(operations), len(caches))


if __name__ == "__main__":
    main()
