"""Versioned execution bounds and separately identified measured traffic."""
from __future__ import annotations

from dataclasses import asdict, dataclass

ENVELOPE_VERSION = "native-workload-envelope-v2"
EVIDENCE_VERSION = "native-inference-qualification-v2"


@dataclass(frozen=True)
class QualificationEnvelope:
    context_tokens: int
    output_tokens: int
    slots: int
    chunk_tokens: int
    page_tokens: int
    cache_type: str
    admission_policy: str
    max_scheduled_tokens: int
    max_prefill_tokens: int
    queue_size: int
    timeout_s: float
    output_buffer_events: int
    output_stall_timeout_s: float
    cache_capacity_bytes: int | None
    prefix_cache_budget_bytes: int
    pressure: dict
    kernel_version: str
    arena_version: str
    workload: dict
    ttft_budget_ms: float | None
    decode_gap_budget_ms: float | None
    max_queue_wait_ms: float | None
    max_service_gap_ms: float | None
    schema_version: str = ENVELOPE_VERSION

    def __post_init__(self):
        if self.schema_version != ENVELOPE_VERSION:
            raise ValueError("legacy envelopes are diagnostic only")
        for name in ("context_tokens", "output_tokens", "slots", "chunk_tokens", "page_tokens",
                     "max_scheduled_tokens", "max_prefill_tokens", "queue_size",
                     "output_buffer_events"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"invalid qualification bound: {name}")
        if (not self.workload or not self.pressure or self.timeout_s < 0
                or self.output_stall_timeout_s <= 0):
            raise ValueError("incomplete qualification envelope")

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or value.get("schema_version") != ENVELOPE_VERSION:
            raise ValueError("current complete workload envelope is required")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("incomplete qualification envelope") from exc

    def to_dict(self):
        return asdict(self)

    def execution_bounds(self):
        result = self.to_dict()
        for key in ("workload", "ttft_budget_ms", "decode_gap_budget_ms", "max_queue_wait_ms",
                    "max_service_gap_ms"):
            result.pop(key)
        return result


def profile_envelope(profile):
    from mlx_one.engine.arena import ARENA_VERSION
    from mlx_one.engine.paged import KERNEL_VERSION
    from mlx_one.engine.pressure import MemoryPressureConfig
    from mlx_one.traffic_replay import generated_traffic, traffic_identity

    traffic = profile.traffic_identity
    if traffic is None and profile.workload in {"mixed", "turnover", "cancellation",
                                               "slow_consumer", "multi_turn", "soak"}:
        traffic = traffic_identity(generated_traffic(profile))
    return QualificationEnvelope(
        profile.context_tokens, profile.output_tokens, profile.slots, profile.chunk_tokens,
        profile.page_tokens, profile.cache_type, profile.admission_policy,
        profile.max_scheduled_tokens, profile.max_prefill_tokens,
        profile.queue_size or max(profile.slots * 4, 8), profile.timeout_seconds,
        profile.output_buffer_events, profile.output_stall_timeout_s,
        profile.cache_capacity_bytes, profile.prefix_cache_budget_bytes,
        profile.pressure_config or asdict(MemoryPressureConfig()), KERNEL_VERSION, ARENA_VERSION,
        {"name": profile.workload, "traffic_sha256": traffic, "seed": profile.seed,
         "shared_prefix_fraction": profile.shared_prefix_fraction,
         "prefix_reuse": profile.workload in {"prefix", "multi_turn", "soak"},
         "dataset": profile.quality_dataset_identity},
        profile.ttft_budget_ms, profile.decode_gap_budget_ms,
        profile.max_queue_wait_ms, profile.max_service_gap_ms)
