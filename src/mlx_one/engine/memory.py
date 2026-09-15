"""Unified-memory forecasting and deterministic request admission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AdmissionStatus(str, Enum):
    ADMIT = "admit"
    QUEUE = "queue"
    REJECT = "reject"


@dataclass(frozen=True)
class MemoryBudget:
    total_bytes: int
    process_bytes: int
    os_reserve_bytes: int
    runtime_reserve_bytes: int

    @classmethod
    def from_total(
        cls, total_bytes: int, *, process_bytes: int | None = None
    ) -> MemoryBudget:
        if total_bytes < 1:
            raise ValueError("total unified memory must be positive")
        os_reserve = max(4 * 1024**3, int(total_bytes * 0.20))
        available = max(total_bytes - os_reserve, 0)
        process = min(process_bytes, available) if process_bytes is not None else available
        if process < 1:
            raise ValueError("unified-memory process budget is empty")
        runtime_reserve = max(512 * 1024**2, int(process * 0.05))
        return cls(total_bytes, process, os_reserve, runtime_reserve)


@dataclass(frozen=True)
class MemoryUsage:
    model_bytes: int = 0
    draft_model_bytes: int = 0
    active_cache_bytes: int = 0
    prefix_cache_bytes: int = 0
    multimodal_bytes: int = 0
    active_mlx_bytes: int = 0

    @property
    def accounted_bytes(self) -> int:
        components = (
            self.model_bytes,
            self.draft_model_bytes,
            self.active_cache_bytes,
            self.prefix_cache_bytes,
            self.multimodal_bytes,
        )
        return max(sum(components), self.active_mlx_bytes)


@dataclass(frozen=True)
class AdmissionDecision:
    status: AdmissionStatus
    requested_bytes: int
    available_bytes: int
    reason: str | None = None


class UnifiedMemoryPlanner:
    def __init__(self, budget: MemoryBudget) -> None:
        self.budget = budget

    def available_bytes(self, usage: MemoryUsage) -> int:
        return max(
            self.budget.process_bytes
            - self.budget.runtime_reserve_bytes
            - usage.accounted_bytes,
            0,
        )

    def admit(
        self,
        requested_bytes: int,
        usage: MemoryUsage,
        *,
        queue_when_busy: bool = True,
    ) -> AdmissionDecision:
        if requested_bytes < 0:
            raise ValueError("requested memory cannot be negative")
        available = self.available_bytes(usage)
        if requested_bytes <= available:
            return AdmissionDecision(AdmissionStatus.ADMIT, requested_bytes, available)
        empty_available = self.available_bytes(MemoryUsage())
        if requested_bytes > empty_available:
            return AdmissionDecision(
                AdmissionStatus.REJECT,
                requested_bytes,
                available,
                "request exceeds the configured unified-memory capacity",
            )
        return AdmissionDecision(
            AdmissionStatus.QUEUE if queue_when_busy else AdmissionStatus.REJECT,
            requested_bytes,
            available,
            "request does not fit while current engine allocations are active",
        )


def forecast_kv_bytes(
    config: Any,
    tokens: int,
    *,
    batch_size: int = 1,
    key_bits: int = 16,
    value_bits: int = 16,
) -> int:
    """Forecast attention KV storage without counting recurrent/conv states."""

    if tokens < 0 or batch_size < 1:
        raise ValueError("KV forecast tokens and batch size must be valid")
    if key_bits not in {4, 8, 16} or value_bits not in {4, 8, 16}:
        raise ValueError("KV forecast supports 4, 8, or 16 bits")
    text = getattr(config, "text_config", config)
    layer_count = _positive_int(text, "num_hidden_layers", "n_layer", "num_layers")
    layer_types = getattr(text, "layer_types", None)
    if isinstance(layer_types, (tuple, list)):
        attention_layers = sum("attention" in str(kind) for kind in layer_types)
    else:
        attention_layers = layer_count
    heads = _positive_int(
        text,
        "num_key_value_heads",
        "num_kv_heads",
        "n_head",
        "num_attention_heads",
    )
    head_dim = getattr(text, "head_dim", None)
    if not isinstance(head_dim, int) or head_dim < 1:
        hidden = _positive_int(text, "hidden_size", "n_embd", "model_dim")
        query_heads = _positive_int(text, "num_attention_heads", "n_head")
        head_dim = hidden // query_heads
    key_bytes = (heads * head_dim * key_bits + 7) // 8
    value_bytes = (heads * head_dim * value_bits + 7) // 8
    return batch_size * tokens * attention_layers * (key_bytes + value_bytes)


def _positive_int(value: Any, *names: str) -> int:
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    raise ValueError(f"model config is missing a positive {'/'.join(names)}")
