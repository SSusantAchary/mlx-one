"""Native attention dispatch and forced-fused diagnostics with stable reference fallback."""
from __future__ import annotations

from functools import lru_cache
from importlib.metadata import version

from mlx_one.engine.kernel_dispatch import KernelDecision, sdpa_decision
from mlx_one.engine.optimization_policy import current_math
from mlx_one.ops.profiling import dispatch, phase


def validate_attention_inputs(q, k, v, *, mask=None, history=0):
    """Validate before cache append, including reference/fallback configurations."""
    if any(len(t.shape) != 4 or any(dim < 1 for dim in t.shape) for t in (q, k, v)):
        raise ValueError("attention requires positive [B,H,T,D] tensors")
    b, hq, tq, d = q.shape
    bk, hk, tk, dk = k.shape
    if (b != bk or v.shape[:3] != k.shape[:3] or d != dk or hq % hk
            or not isinstance(history, int) or history < 0):
        raise ValueError("attention batch/head/GQA/width/history layout mismatch")
    if isinstance(mask, str):
        if mask != "causal" or tq > tk + history:
            raise ValueError("unsupported attention mask or causal alignment")
    elif mask is not None:
        expected = (b, hq, tq, tk + history)
        if mask.ndim > 4 or any(a not in (1, size) for a, size in
                zip((1,) * (4 - mask.ndim) + tuple(mask.shape), expected, strict=True)):
            raise ValueError("attention mask cannot broadcast to actual query/key layout")


def attention_reference(q, k, v, *, scale, mask=None, stable=False, promote=False):
    validate_attention_inputs(q, k, v, mask=mask)
    import mlx.core as mx

    from mlx_one.models.shared.execution import attention_matmul

    operand_dtype = q.dtype
    accumulate_fp32 = not stable and str(q.dtype).rsplit(".", 1)[-1] in {
        "float16", "bfloat16"}
    if accumulate_fp32:
        # Preserve operand precision while avoiding rounded low-precision scores
        # before softmax. This is reference arithmetic, not optimized dispatch.
        q, k, v = (x.astype(mx.float32) for x in (q, k, v))
    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
        dispatch("kv_head_repeat", "reference", "reference", "materialized GQA head repeats",
                 bytes=(k.nbytes + v.nbytes) * repeats)
        k, v = mx.repeat(k, repeats, axis=1), mx.repeat(v, repeats, axis=1)
    scores = (attention_matmul(q, k.transpose(0, 1, 3, 2)) if stable else
              q @ k.transpose(0, 1, 3, 2)) * scale
    if promote:
        scores = scores.astype(mx.float32)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError("unsupported attention mask")
        # Causal alignment is at the lower-right, including a committed offset.
        offset = k.shape[2] - q.shape[2]
        mask = mx.triu(mx.full((q.shape[2], k.shape[2]), -float("inf")), k=offset + 1)
    if mask is not None:
        scores = (mx.where(mask, scores, -float("inf"))
                  if str(mask.dtype).rsplit(".", 1)[-1] in {"bool", "bool_"}
                  else scores + mask.astype(scores.dtype))
    probabilities = mx.softmax(scores, axis=-1, precise=True)
    result = attention_matmul(probabilities, v) if stable else probabilities @ v
    return result.astype(operand_dtype) if accumulate_fp32 and not promote else result


@lru_cache(maxsize=1)
def mlx_version():
    return tuple(int(p) for p in version("mlx").split(".")[:3])


def attention_fused(q, k, v, *, scale, mask=None, accumulation="float32"):
    """Execute actual fused Metal or raise; this function grants no qualification.

    Runtime dispatch retains its existing FP32-operand experiment. Native-dtype
    calls are for supervised numerical/performance diagnostics only. Neither
    variant repeats KV heads or retries a failed kernel through reference math.
    """
    validate_attention_inputs(q, k, v, mask=mask)
    import mlx.core as mx

    if accumulation not in {"float32", "native"}:
        raise ValueError("unsupported fused accumulation policy")
    dtypes = tuple(str(t.dtype).rsplit(".", 1)[-1] for t in (q, k, v))
    if (len(set(dtypes)) != 1 or dtypes[0] not in {"float16", "bfloat16", "float32"}
            or q.shape[-1] not in {64, 128, 256} or v.shape[-1] != q.shape[-1]):
        raise ValueError("unsupported fused Q/K/V dtype or head dimensions")
    if (mlx_version() < (0, 32) or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or not callable(getattr(getattr(mx, "fast", None),
                                    "scaled_dot_product_attention", None))):
        raise ValueError("requires MLX >=0.32 with active Metal fused attention")
    if mask is not None and not isinstance(mask, str):
        kind = str(mask.dtype).rsplit(".", 1)[-1]
        if kind not in {"bool", "bool_", "float16", "bfloat16", "float32"}:
            raise ValueError("unsupported fused mask dtype")
        if kind not in {"bool", "bool_"}:
            if accumulation == "float32":
                mask = mask.astype(mx.float32)
            elif kind != dtypes[0]:
                raise ValueError("native fused additive mask must preserve operand precision")
    if accumulation == "float32":
        q, k, v = (t.astype(mx.float32) for t in (q, k, v))
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, force_fused=True)


STABLE_ROUNDING_REASON = (
    "stable INT8 FP16 rounding requires reference: MLX 0.32.2 fused FP32 reduction/"
    "online-softmax differences cross the output cast boundary and failed Qwen3 state parity"
)


NATIVE_GATHER_BACKEND = "native_gather_tiled"
NATIVE_GATHER_TILE = 128


@lru_cache(maxsize=1)
def _native_gather_device_name():
    import mlx.core as mx

    device_info = getattr(mx, "device_info", None)
    return device_info().get("device_name", "") if callable(device_info) else ""


def _native_gather_decision(q, k, v, *, mask, history, requires_stable_rounding):
    """Measured availability envelope; checkpoint qualification stays separate."""
    import mlx.core as mx

    context = current_math()

    def fallback(reason):
        return KernelDecision("reference", f"{NATIVE_GATHER_BACKEND} fallback: {reason}")

    if context.policy.attention != NATIVE_GATHER_BACKEND:
        return fallback("explicit native_gather_tiled policy required")
    if mlx_version() != (0, 32, 2):
        return fallback("requires exact MLX 0.32.2")
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return fallback("requires active Metal GPU")
    if _native_gather_device_name() != "Apple M4":
        return fallback("requires measured Apple M4 device")
    if context.family != "qwen3" or not requires_stable_rounding:
        return fallback("requires Qwen3 stable INT8 FP16 execution")
    dtypes = tuple(str(t.dtype).rsplit(".", 1)[-1] for t in (q, k, v))
    if dtypes != ("float16",) * 3:
        return fallback("requires actual FP16 Q/K/V")
    batch, heads, queries, width = q.shape
    if ((batch, heads, width) != (1, 16, 128)
            or k.shape[:2] != (1, 8) or k.shape[-1] != 128
            or v.shape != k.shape or queries not in {1, 7, 64}
            or not queries <= k.shape[2] + history <= 96):
        return fallback("outside B1 Hq16/Hkv8 D128 Q{1,7,64} Q<=K<=96 envelope")
    if mask is not None and (isinstance(mask, str)
            or str(mask.dtype).rsplit(".", 1)[-1] != "float32"):
        return fallback("requires no mask or the original FP32 array mask")
    return KernelDecision(NATIVE_GATHER_BACKEND,
        "M4 MLX 0.32.2 stable INT8 FP16 native gather/softmax; tile=128; "
        "checkpoint qualification separate")


def select_attention(q, k, v, *, mask=None, history=0, requires_stable_rounding=False):
    validate_attention_inputs(q, k, v, mask=mask, history=history)
    import mlx.core as mx

    context = current_math()
    if context.policy.attention == "reference":
        return KernelDecision("reference", "reference policy")
    if context.policy.attention == NATIVE_GATHER_BACKEND:
        return _native_gather_decision(q, k, v, mask=mask, history=history,
                                      requires_stable_rounding=requires_stable_rounding)
    if requires_stable_rounding:
        # This decision happens before cache append. Operator-level FP32 parity
        # does not establish the FP16 rounding contract of stable INT8 decoders.
        return KernelDecision("reference", STABLE_ROUNDING_REASON)
    if mask is not None and not isinstance(mask, str):
        kind = str(mask.dtype).rsplit(".", 1)[-1]
        if kind not in {"bool", "bool_", "float16"}:
            return KernelDecision("reference", "additive mask precision must be preserved")
    k_shape, v_shape = list(k.shape), list(v.shape)
    k_shape[2] += history
    v_shape[2] += history
    return sdpa_decision(context.policy, family=context.family,
        q_shape=q.shape, k_shape=tuple(k_shape), v_shape=tuple(v_shape),
        dtypes=tuple(str(t.dtype).rsplit(".", 1)[-1] for t in (q, k, v)),
        mask_kind="none" if mask is None else mask if isinstance(mask, str) else "array",
        mlx_version=mlx_version(),
        metal_available=mx.metal.is_available() and mx.default_device() == mx.gpu)


def attention(q, k, v, *, scale, mask=None, stable=False, promote=False, decision=None):
    validate_attention_inputs(q, k, v, mask=mask)
    context = current_math()
    requires_stable_rounding = stable and str(q.dtype).rsplit(".", 1)[-1] == "float16"
    decision = decision or select_attention(q, k, v, mask=mask,
                                            requires_stable_rounding=requires_stable_rounding)
    if decision.selected == NATIVE_GATHER_BACKEND:
        # A supplied decision must meet the same live capability envelope. This
        # static check precedes candidate execution; executed errors never retry.
        decision = _native_gather_decision(q, k, v, mask=mask, history=0,
                                          requires_stable_rounding=requires_stable_rounding)
        if decision.selected == NATIVE_GATHER_BACKEND and scale != 128**-0.5:
            decision = KernelDecision("reference",
                "native_gather_tiled fallback: scale outside measured D128 workload")
    if requires_stable_rounding and decision.selected == "mlx_sdpa":
        # Supplied decisions cannot bypass a stable arithmetic contract. Forced
        # qualification calls attention_fused directly and records that route.
        decision = KernelDecision("reference", STABLE_ROUNDING_REASON)
    dispatch("attention", context.policy.attention, decision.selected, decision.reason,
             q=list(q.shape), k=list(k.shape), phase=context.phase,
             storage_dtypes=[str(k.dtype), str(v.dtype)],
             accumulation_operands="float32"
             if decision.selected in {"mlx_sdpa", NATIVE_GATHER_BACKEND} else "reference",
             native_sdpa=decision.selected == "mlx_sdpa",
             query_tile=NATIVE_GATHER_TILE if decision.selected == NATIVE_GATHER_BACKEND else None)
    result = None
    with phase("attention_core", output=lambda: result):
        if decision.selected == NATIVE_GATHER_BACKEND:
            from mlx_one.ops.reference_tiled import reference_compatible_attention

            result = reference_compatible_attention(q, k, v, scale=scale, mask=mask,
                                                   query_tile=NATIVE_GATHER_TILE)
        elif decision.selected == "mlx_sdpa":
            # No repeat/gather of native GQA KV heads. Failure propagates to the
            # owning evaluated cache transaction; never retry a mutated cache.
            # Match the reference's FP32 attention arithmetic. FP16 fused
            # reductions failed real nonuniform-prompt state parity. Cache
            # storage remains FP16; these are bounded transient operands.
            result = attention_fused(q, k, v, scale=scale, mask=mask, accumulation="float32")
        else:
            result = attention_reference(q, k, v, scale=scale, mask=mask,
                                         stable=stable, promote=promote)
    return result
