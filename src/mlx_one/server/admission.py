"""Unified-memory request admission for the native server."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mlx_one.engine.memory import (
    AdmissionDecision,
    MemoryBudget,
    MemoryUsage,
    UnifiedMemoryPlanner,
    forecast_kv_bytes,
)
from mlx_one.hardware import detect_hardware
from mlx_one.utils.memory import get_memory_snapshot


class MemoryAdmission:
    def __init__(self, budget: MemoryBudget) -> None:
        self.planner = UnifiedMemoryPlanner(budget)

    @classmethod
    def detect(cls) -> MemoryAdmission | None:
        hardware = detect_hardware()
        if hardware.memory_bytes is None:
            return None
        return cls(
            MemoryBudget.from_total(
                hardware.memory_bytes,
                process_bytes=hardware.process_memory_budget_bytes,
            )
        )

    def decide(
        self,
        bundle: Any,
        tokens: int,
        *,
        key_bits: int = 16,
        value_bits: int = 16,
        prefix_cache_bytes: int = 0,
        multimodal_bytes: int = 0,
    ) -> AdmissionDecision:
        requested = forecast_kv_bytes(
            bundle.model.config,
            tokens,
            key_bits=key_bits,
            value_bits=value_bits,
        )
        snapshot = get_memory_snapshot()
        usage = MemoryUsage(
            prefix_cache_bytes=prefix_cache_bytes,
            multimodal_bytes=multimodal_bytes,
            active_mlx_bytes=snapshot.active_bytes,
        )
        return self.planner.admit(requested, usage)

    def stats(self) -> dict[str, int]:
        return asdict(self.planner.budget)
