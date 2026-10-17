"""Use stock MLX packed affine multiplication only for validated actual call shapes."""
from __future__ import annotations

import math

from mlx_one.engine.kernel_dispatch import qmm_decision
from mlx_one.engine.optimization_policy import current_math
from mlx_one.ops.profiling import dispatch


def try_quantized_linear(module, value, *, as_linear=False):
    context = current_math()
    if context.policy.projection == "reference":
        return False, None
    import mlx.core as mx

    from mlx_one.ops.attention import mlx_version

    decision = qmm_decision(context.policy, family=context.family,
        mlx_version=mlx_version(),
        metal_available=mx.metal.is_available() and mx.default_device() == mx.gpu,
        rows=math.prod(value.shape[:-1]), width=value.shape[-1],
        weight_shape=getattr(getattr(module, "weight", None), "shape", ()),
        scales_shape=getattr(getattr(module, "scales", None), "shape", ()),
        biases_shape=getattr(getattr(module, "biases", None), "shape", ()),
        bits=getattr(module, "bits", None), group_size=getattr(module, "group_size", None),
        mode=getattr(module, "mode", None),
        weight_dtype=str(getattr(getattr(module, "weight", None), "dtype", "")).rsplit(".", 1)[-1],
        activation_dtype=str(value.dtype).rsplit(".", 1)[-1],
        metadata_dtypes=tuple(str(getattr(getattr(module, name, None), "dtype", "")).rsplit(
            ".", 1)[-1] for name in ("scales", "biases")))
    dispatch("projection", context.policy.projection, decision.selected, decision.reason,
             rows=math.prod(value.shape[:-1]), width=value.shape[-1], phase=context.phase,
             activation_dtype=str(value.dtype), accumulation_operands="stock",
             output_dtype=str(value.dtype))
    if decision.selected != "mlx_qmm":
        return False, None
    # Unqualified matrix shapes were rejected before execution. The selected
    # vector route preserves native FP16 inputs and original packed parameters.
    result = mx.quantized_matmul(value, module.weight,
                                 scales=module.scales, biases=module.biases,
                                 transpose=True, group_size=module.group_size, bits=module.bits,
                                 mode="affine")
    if not as_linear and "bias" in module:
        result = result + module.bias
    return True, result
