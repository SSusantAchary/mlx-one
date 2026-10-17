"""Pure shape/layout checks used before native execution can mutate request state."""
from __future__ import annotations

from dataclasses import dataclass

MATH_KERNEL_VERSION = "native-math-kernels-v7"
SUPPORTED_FAMILIES = frozenset({"qwen2", "qwen3", "lfm2", "qwen3_5", "gpt2"})


@dataclass(frozen=True)
class KernelDecision:
    selected: str
    reason: str


def math_availability(family, *, quantized, configured_fp16):
    """Implementation/availability are distinct from unmeasured qualification."""
    supported = family in SUPPORTED_FAMILIES
    def record(implemented, available, reason):
        return {"implemented": implemented,
                "availability": "experimental" if available else "unavailable",
                "qualification": "unqualified", "reason": reason}
    return {
        "last_logits": record(supported, supported and configured_fp16,
            "explicit dense B1 FP16 policy; per-checkpoint performance remains unqualified"),
        "packed_vector": record(supported, supported and quantized and configured_fp16,
            "validated affine packed layout and actual call rows=1 required"),
        "packed_matrix": record(True, False,
            "stock FP16/FP32 matrix candidates failed native full-state parity"),
        "fused_attention": record(family in {"qwen2", "qwen3"}, False,
            "Qwen state parity failed; other component dispatch is not validated"),
        "hybrid_fast": record(False, False, "no validated fused hybrid transition"),
        "compiled_engine": record(False, False,
            "pure-op experiment only; native graph memory pending"),
    }


def sdpa_decision(policy, *, family, q_shape, k_shape, v_shape, dtypes, mask_kind,
                  mlx_version, metal_available):
    if policy.attention == "reference":
        return KernelDecision("reference", "reference policy")
    if policy.attention != "mlx_sdpa":
        return KernelDecision("reference", "custom attention route has no measured qualification")
    if family not in SUPPORTED_FAMILIES:
        return KernelDecision("reference", "unsupported model family")
    if family in {"qwen2", "qwen3"}:
        return KernelDecision("reference",
            "native Qwen SDPA prefill/decode failed nonuniform state parity; retain reference")
    if not metal_available or tuple(mlx_version[:2]) < (0, 32):
        return KernelDecision("reference", "requires tested MLX Metal SDPA interface")
    if any(len(shape) != 4 for shape in (q_shape, k_shape, v_shape)):
        raise ValueError("attention requires [B,H,T,D] tensors")
    b, hq, tq, d = q_shape
    bk, hk, tk, dk = k_shape
    if (b != 1 or bk != b or v_shape != k_shape or d != dk or min(hq, hk, tq, tk) < 1
            or hq % hk or d not in {64, 128, 256} or tq > tk):
        return KernelDecision("reference", "unsupported B1 GQA/head/length layout")
    if len(set(dtypes)) != 1 or dtypes[0] != "float16":
        return KernelDecision("reference", "requires actual FP16 Q/K/V")
    if tq == 1:
        return KernelDecision("reference",
            "native single-query/decode SDPA failed FP16 state parity; retain reference")
    if mask_kind not in {"none", "causal", "array"}:
        return KernelDecision("reference", "unsupported attention mask")
    return KernelDecision("mlx_sdpa", "experimental fused FP32 operands over FP16 KV; unqualified")


def qmm_decision(policy, *, family, rows, width, weight_shape, scales_shape, biases_shape,
                 bits, group_size, mode, weight_dtype, activation_dtype, metadata_dtypes,
                 mlx_version, metal_available):
    if policy.projection == "reference":
        return KernelDecision("reference", "reference policy")
    if policy.projection != "mlx_qmm":
        return KernelDecision("reference", "custom projection has no measured qualification")
    if not metal_available or tuple(mlx_version[:2]) < (0, 32):
        return KernelDecision("reference", "requires tested MLX Metal packed QMM interface")
    if family not in SUPPORTED_FAMILIES or type(rows) is not int or rows < 1:
        return KernelDecision("reference", "unsupported model or actual projection row count")
    if rows != 1:
        return KernelDecision("reference",
            "native packed GEMM has not passed full-state parity; retain rowwise reference")
    if (type(bits) is not int or type(group_size) is not int or type(width) is not int
            or bits not in {4, 8} or group_size not in {32, 64, 128} or mode != "affine"
            or weight_dtype != "uint32" or activation_dtype != "float16"
            or len(set(metadata_dtypes)) != 1 or metadata_dtypes[0] != "float16"):
        return KernelDecision("reference", "unsupported packed affine precision/layout")
    if (width % group_size or len(weight_shape) != 2
            or weight_shape[-1] * (32 // bits) != width
            or scales_shape != (weight_shape[0], width // group_size)
            or biases_shape != scales_shape):
        return KernelDecision("reference", "packed weight/affine group shape mismatch")
    return KernelDecision("mlx_qmm", "stock packed GEMV; actual input rows=1")
