"""Host-side segmentation results; confidence and predicted IoU remain distinct."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConceptSegmentationResult:
    masks: Any
    scores: Any
    boxes: Any
    original_size: tuple[int, int]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractiveSegmentationResult:
    masks: Any
    predicted_iou: Any
    boxes: Any
    original_size: tuple[int, int]
    low_res_logits: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoFrameResult(ConceptSegmentationResult):
    frame_index: int = 0
    object_ids: tuple[int, ...] = ()
    prompt_ids: tuple[int, ...] = ()
