"""Immutable, independently reversible B1 experiments; no runtime imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MLXBoostConfig:
    # Skip calibration identity construction only when no candidate exists.
    # Model/evaluator/sampler/cache execution order is unchanged.
    fast_decode: bool = False
    gptoss_prefill_sort: bool = False
    gptoss_mlx_lm: bool = False
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
        model_config=None,
        device_info=None,
        cache_types=None,
    ):
        unsupported = [
            name
            for name, value in asdict(self).items()
            if value
            and name
            not in {"fast_decode", "gptoss_prefill_sort", "gptoss_mlx_lm", "strict_reference"}
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
        gptoss = self.gptoss_prefill_sort or self.gptoss_mlx_lm
        if gptoss:
            from mlx_one.mlxboost.gptoss_profile import supports_prefill_sort

            eligible = (
                not self.fast_decode
                and not (self.gptoss_prefill_sort and self.gptoss_mlx_lm)
                and supports_prefill_sort(
                    family, dtype, quantization, model_config, device_info, cache_types
                )
            )
        reason = (
            "unimplemented/unqualified switches: " + ", ".join(unsupported)
            if unsupported
            else "requires GPT-OSS-20B MXFP4/Q4 BF16 on M4 32 GB, matched floating KV, "
            "fast_decode disabled, and one GPT-OSS recipe at a time"
            if gptoss and not eligible
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
