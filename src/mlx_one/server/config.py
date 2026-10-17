"""Immutable native server configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal, cast

from mlx_one.engine.modes import InferenceMode, ModelCapabilityUnsupported, resolve_backend
from mlx_one.engine.pressure import MemoryPressureConfig

CacheType = Literal["f16", "bf16", "q4_0", "q8_0"]
CacheBackend = Literal["dense", "block", "paged", "hybrid"]


@dataclass(frozen=True)
class ServerConfig:
    alias: str | None = None
    api_keys: tuple[str, ...] = ()
    context_length: int | None = None
    queue_size: int = 8
    timeout: float = 600.0
    warmup: bool = True
    parallel: int = 1
    prefill_chunk_size: int | None = 512
    max_batch_tokens: int | None = 2048
    prefix_cache_bytes: int = 512 * 1024**2
    cache_backend: CacheBackend | None = None
    inference_mode: InferenceMode | None = None
    execution_dtype: str | None = None
    admission_policy: Literal["deadline", "cache_aware"] = "deadline"
    memory_pressure_config: MemoryPressureConfig | None = None
    qualification_evidence: tuple[str, ...] = ()
    qualification_profile: dict[str, Any] | None = None
    ttft_budget_ms: float | None = None
    decode_gap_budget_ms: float | None = None
    max_prefill_tokens: int = 1024
    cache_block_size: int = 32
    cache_memory_budget_bytes: int | None = None
    reasoning: Literal["auto", "on", "off"] = "auto"
    reasoning_format: Literal["none", "deepseek", "deepseek-legacy"] = "deepseek"
    reasoning_budget: int = -1
    reasoning_preserve: bool = False
    cache_prompt: bool = True
    cache_reuse: int = 256
    cache_idle_slots: bool = False
    context_shift: bool = False
    cache_type_k: CacheType = "f16"
    cache_type_v: CacheType = "f16"
    spec_type: Literal["none", "draft-mtp"] = "none"
    spec_draft_n_max: int = 3
    transcription_enabled: bool = False

    def __post_init__(self) -> None:
        if self.inference_mode == "native_optimized":
            raise ModelCapabilityUnsupported(
                "native_optimized math policy currently requires the Python Engine")
        object.__setattr__(
            self, "cache_backend", resolve_backend(self.inference_mode, self.cache_backend)
        )
        if self.execution_dtype not in {None, "float16", "bfloat16"}:
            raise ValueError("execution dtype must be float16 or bfloat16")
        if self.admission_policy not in {"deadline", "cache_aware"}:
            raise ValueError("unsupported admission policy")
        if self.admission_policy == "cache_aware" and self.inference_mode not in {
            "experimental_dense_batch", "paged", "experimental_hybrid_batch", "auto"
        }:
            raise ValueError("cache-aware admission requires an actor execution mode")
        if ((self.ttft_budget_ms is None) != (self.decode_gap_budget_ms is None)
                or any(not isfinite(v) or v <= 0 for v in
                       (self.ttft_budget_ms, self.decode_gap_budget_ms) if v is not None)):
            raise ValueError("declare both finite positive service budgets")
        if self.inference_mode in {"paged", "experimental_hybrid_batch", "auto"} and (
            self.spec_type != "none" or self.cache_type_k != self.cache_type_v
        ):
            raise ModelCapabilityUnsupported(
                "new execution modes require matched KV, no speculation")
        if self.max_prefill_tokens < 1:
            raise ValueError("max prefill tokens must be positive")
        if self.inference_mode == "experimental_dense_batch" and (
            self.cache_type_k != "f16" or self.cache_type_v != "f16" or self.spec_type != "none"
        ):
            raise ModelCapabilityUnsupported(
                "dense batching requires FP16 KV and speculation disabled")
        if self.alias is not None and not self.alias.strip():
            raise ValueError("alias cannot be empty")
        if any(not key for key in self.api_keys):
            raise ValueError("API keys cannot be empty")
        if self.context_length is not None and self.context_length < 1:
            raise ValueError("context length must be positive")
        if self.queue_size < 1 or self.parallel < 1:
            raise ValueError("queue size and parallel slots must be positive")
        if self.prefill_chunk_size is not None and self.prefill_chunk_size < 1:
            raise ValueError("prefill chunk size must be positive or auto")
        if self.max_batch_tokens is not None and self.max_batch_tokens < 1:
            raise ValueError("maximum batch tokens must be positive or auto")
        if self.prefix_cache_bytes < 0:
            raise ValueError("prefix cache capacity cannot be negative")
        if self.inference_mode == "experimental_block" and (
            self.cache_type_k != "f16" or self.cache_type_v != "f16"
        ):
            raise ModelCapabilityUnsupported(
                "experimental_block currently requires f16 KV settings"
            )
        if self.cache_backend not in {"dense", "block", "paged", "hybrid"}:
            raise ValueError("cache backend must be dense or block")
        if self.cache_block_size not in {16, 32, 64, 128}:
            raise ValueError("cache block size must be 16, 32, 64, or 128")
        if self.cache_memory_budget_bytes is not None and self.cache_memory_budget_bytes < 1:
            raise ValueError("cache memory budget must be positive")
        if self.timeout < 0:
            raise ValueError("timeout cannot be negative")
        if self.reasoning_budget < -1:
            raise ValueError("reasoning budget must be -1 or greater")
        if self.cache_reuse < 0:
            raise ValueError("cache reuse cannot be negative")
        if self.spec_draft_n_max < 1:
            raise ValueError("speculative draft count must be positive")
        supported_cache_types = {"f16", "bf16", "q4_0", "q8_0"}
        if (
            self.cache_type_k not in supported_cache_types
            or self.cache_type_v not in supported_cache_types
        ):
            raise ValueError("unsupported KV cache type")

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("api_keys", None)
        return result


def cache_type_from_bits(bits: int) -> CacheType:
    return cast(CacheType, {4: "q4_0", 8: "q8_0", 16: "f16"}[bits])
