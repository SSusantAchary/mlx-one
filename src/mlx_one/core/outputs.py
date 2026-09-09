"""Common model outputs shared across native architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelOutput:
    logits: Any
    last_hidden_state: Any
    cache: tuple[Any, ...] | None = None
    hidden_states: tuple[Any, ...] | None = None
    router_logits: tuple[Any, ...] | None = None
    router_aux_loss: Any | None = None


@dataclass
class VisionLanguageModelOutput(ModelOutput):
    vision_hidden_states: Any | None = None
    position_ids: Any | None = None
    rope_deltas: Any | None = None
