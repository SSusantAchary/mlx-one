"""B1 attention dispatch preserving the original reference reduction as fallback."""
from __future__ import annotations

from functools import lru_cache
from importlib.metadata import version

from mlx_one.engine.kernel_dispatch import KernelDecision, sdpa_decision
from mlx_one.engine.optimization_policy import current_math
from mlx_one.ops.profiling import dispatch, phase


def attention_reference(q, k, v, *, scale, mask=None, stable=False, promote=False):
    import mlx.core as mx

    from mlx_one.models.shared.execution import attention_matmul

    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
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
                  if str(mask.dtype).rsplit(".", 1)[-1] in {"bool", "bool_"} else scores + mask)
    probabilities = mx.softmax(scores, axis=-1, precise=True)
    return attention_matmul(probabilities, v) if stable else probabilities @ v


@lru_cache(maxsize=1)
def mlx_version():
    return tuple(int(p) for p in version("mlx").split(".")[:2])


def select_attention(q, k, v, *, mask=None, history=0):
    import mlx.core as mx

    context = current_math()
    if context.policy.attention == "reference":
        return KernelDecision("reference", "reference policy")
    if mask is not None and not isinstance(mask, str):
        expected = (q.shape[0], q.shape[1], q.shape[2], k.shape[2] + history)
        if mask.ndim > 4 or any(a not in (1, b) for a, b in
                               zip((1,) * (4 - mask.ndim) + mask.shape, expected, strict=True)):
            raise ValueError("attention mask cannot broadcast to actual query/key layout")
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
    import mlx.core as mx

    context = current_math()
    decision = decision or select_attention(q, k, v, mask=mask)
    dispatch("attention", context.policy.attention, decision.selected, decision.reason,
             q=list(q.shape), k=list(k.shape), phase=context.phase,
             storage_dtypes=[str(k.dtype), str(v.dtype)],
             accumulation_operands="float32" if decision.selected == "mlx_sdpa" else "reference")
    result = None
    with phase("attention_core", output=lambda: result):
        if decision.selected == "mlx_sdpa":
            # No repeat/gather of native GQA KV heads. Failure propagates to the
            # owning evaluated cache transaction; never retry a mutated cache.
            # Match the reference's FP32 attention arithmetic. FP16 fused
            # reductions failed real nonuniform-prompt state parity. Cache
            # storage remains FP16; these are bounded transient operands.
            if mask is not None and not isinstance(mask, str):
                if str(mask.dtype).rsplit(".", 1)[-1] not in {"bool", "bool_"}:
                    mask = mask.astype(mx.float32)
            result = mx.fast.scaled_dot_product_attention(
                q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=scale,
                                                         mask=mask, force_fused=True)
        else:
            result = attention_reference(q, k, v, scale=scale, mask=mask,
                                         stable=stable, promote=promote)
    return result
