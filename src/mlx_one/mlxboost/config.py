"""Immutable, independently reversible B1 experiments; no runtime imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MLXBoostConfig:
    # Skip calibration identity construction only when no candidate exists.
    # Model/evaluator/sampler/cache execution order is unchanged.
    fast_decode: bool = False
    native_quantized_dispatch: bool = False
    dense_cache_capacity: bool = False
    fused_attention: bool = False
    stream_bridge: bool = False
    compile_experiment: bool = False
    experimental_qgemv: bool = False
    strict_reference: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise TypeError("MLXBoost switches must be boolean")

    @property
    def enabled(self):
        return any(value for key, value in asdict(self).items() if key != "strict_reference")

    def resolve(
        self,
        *,
        family,
        dtype,
        quantization,
        backend,
        slots,
        cache_reuse,
        mlx_version,
        metal_available,
    ):
        unsupported = [
            name
            for name, value in asdict(self).items()
            if value and name not in {"fast_decode", "strict_reference"}
        ]
        eligible = (
            family == "qwen2"
            and dtype == "float16"
            and quantization.get("bits") == 8
            and quantization.get("group_size") == 64
            or family == "qwen3"
            and dtype == "bfloat16"
            and quantization.get("bits") == 8
            and quantization.get("group_size") == 64
            or family == "llama"
            and dtype == "bfloat16"
            and not quantization
        )
        reason = (
            "unimplemented/unqualified switches: " + ", ".join(unsupported)
            if unsupported
            else "unsupported model/precision/quantization"
            if not eligible
            else "requires dense B1, cache reuse off, MLX 0.32.2 and Metal"
            if backend != "dense"
            or slots != 1
            or cache_reuse
            or mlx_version != "0.32.2"
            or not metal_available
            else None
        )
        if reason:
            if self.strict_reference:
                raise ValueError("MLXBoost strict_reference: " + reason)
            return MLXBoostConfig(), reason
        return self, None
