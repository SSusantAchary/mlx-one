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


@dataclass
class EmbeddingModelOutput:
    embeddings: Any
    last_hidden_state: Any
    attention_mask: Any | None = None
    pooler_output: Any | None = None
    hidden_states: tuple[Any, ...] | None = None


@dataclass
class AudioModelOutput:
    text_logits: Any
    audio_logits: Any
    last_hidden_state: Any
    cache: tuple[Any, ...] | None = None
    audio_lengths: Any | None = None
    waveform: Any | None = None


@dataclass
class ASRModelOutput:
    logits: Any
    encoder_last_hidden_state: Any
    decoder_last_hidden_state: Any
    cache: tuple[Any, ...] | None = None
    encoder_hidden_states: tuple[Any, ...] | None = None
    decoder_hidden_states: tuple[Any, ...] | None = None
    decoder_attentions: tuple[Any, ...] | None = None
    cross_attentions: tuple[Any, ...] | None = None
