"""Versioned, bounded text classification contracts. Scores are softmax outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SOURCE_COMMIT = "8cb5963cc22174954e7dca2c0a3320b7dc2f4edc"
MAX_TEXT_BYTES = 32768
MAX_REQUEST_BYTES = 65536
MAX_INPUTS = 2


@dataclass(frozen=True)
class ClassificationResult:
    predictions: tuple[dict, ...]
    model: str
    revision: str
    task: str = "text-classification"
    source_commit: str = SOURCE_COMMIT
    schema_version: str = "mlx-one-classification-v1"
    score_semantics: str = "single-label-softmax"

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


def validate_inputs(texts, max_length=512, top_k=2):
    items = (texts,) if isinstance(texts, str) else texts
    if not isinstance(items, (tuple, list)) or not 1 <= len(items) <= MAX_INPUTS:
        raise ValueError("input must be one nonempty text or a list of one or two texts")
    if any(not isinstance(text, str) or not text.strip() for text in items):
        raise ValueError("input texts must be nonempty and contain non-whitespace characters")
    try:
        sizes = [len(text.encode("utf-8")) for text in items]
    except UnicodeError as exc:
        raise ValueError("input text must be valid Unicode") from exc
    if max(sizes) > MAX_TEXT_BYTES or sum(sizes) > MAX_REQUEST_BYTES:
        raise ValueError("text exceeds 32768 bytes or request exceeds 65536 bytes")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or not 2 <= max_length <= 512
    ):
        raise ValueError(
            "max_length must be an integer from 2 through 512, including special tokens"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k not in (1, 2):
        raise ValueError("top_k must be 1 or 2 for the SST-2 checkpoint")
    return tuple(items)


@dataclass(frozen=True)
class TokenClassificationResult:
    predictions: tuple[dict, ...]
    model: str
    revision: str
    aggregation_strategy: str
    task: str = "token-classification"
    source_commit: str = SOURCE_COMMIT
    schema_version: str = "mlx-one-token-classification-v1"
    offsets: str = "unicode-codepoint-end-exclusive"
    score_semantics: str = "token-softmax-or-source-group-mean"

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class ZeroShotImageResult:
    labels: tuple[dict, ...]
    image_size: dict
    model: str
    revision: str
    task: str = "zero-shot-image-classification"
    source_commit: str = SOURCE_COMMIT
    schema_version: str = "mlx-one-zero-shot-image-v1"
    score_semantics: str = "clip-candidate-softmax"
    hypothesis_template: str = "This is a photo of {}."

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
