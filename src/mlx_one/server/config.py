"""Immutable native server configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

CacheType = Literal["f16", "bf16", "q4_0", "q8_0"]


@dataclass(frozen=True)
class ServerConfig:
    alias: str | None = None
    api_keys: tuple[str, ...] = ()
    context_length: int | None = None
    queue_size: int = 8
    timeout: float = 600.0
    warmup: bool = True
    parallel: int = 1
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

    def __post_init__(self) -> None:
        if self.alias is not None and not self.alias.strip():
            raise ValueError("alias cannot be empty")
        if any(not key for key in self.api_keys):
            raise ValueError("API keys cannot be empty")
        if self.context_length is not None and self.context_length < 1:
            raise ValueError("context length must be positive")
        if self.queue_size < 1 or self.parallel < 1:
            raise ValueError("queue size and parallel slots must be positive")
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
