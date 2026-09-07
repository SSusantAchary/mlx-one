"""Strict local text dataset loading, normalization, and token-mask previews."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mlx_one.schemas import DatasetFormat, DatasetLayout, DatasetSpec


class DatasetError(ValueError):
    """Raised when dataset input is malformed or ambiguous."""


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


@dataclass(frozen=True)
class PreparedRecord:
    """Tokenized causal-LM record with explicit supervised token positions."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    truncated_tokens: int = 0


def dataset_spec_from_path(
    path: str | Path,
    *,
    layout: DatasetLayout | str = DatasetLayout.AUTO,
    split: str = "train",
    response_only: bool = True,
) -> DatasetSpec:
    """Create an immutable local dataset specification including its content hash."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DatasetError(f"dataset does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise DatasetError("dataset must be JSON or JSONL")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return DatasetSpec(
        dataset_id=source.stem,
        source=str(source),
        revision=digest,
        split=split,
        format=DatasetFormat.JSONL if suffix == ".jsonl" else DatasetFormat.JSON,
        layout=layout,
        response_only=response_only,
        sha256=digest,
    )


def load_records(spec: DatasetSpec) -> list[dict[str, Any]]:
    """Read and validate records without executing preprocessing code."""
    path = Path(spec.source).expanduser()
    if not path.is_file():
        raise DatasetError(f"dataset does not exist: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec.sha256:
        raise DatasetError("dataset content does not match DatasetSpec.sha256")
    try:
        if spec.format is DatasetFormat.JSONL:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("data") if isinstance(payload, dict) else payload
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid dataset JSON: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise DatasetError("dataset must contain at least one record")
    if any(not isinstance(item, dict) for item in records):
        raise DatasetError("every dataset record must be an object")
    normalized = [normalize_record(item, spec.layout, spec.columns) for item in records]
    return normalized


def normalize_record(
    record: dict[str, Any],
    layout: DatasetLayout | str = DatasetLayout.AUTO,
    columns: dict[str, str] | Any = None,
) -> dict[str, Any]:
    """Normalize supported layouts to text or prompt/completion records."""
    selected = DatasetLayout(layout)
    names = dict(columns or {})
    if selected is DatasetLayout.AUTO:
        if "messages" in record:
            selected = DatasetLayout.MESSAGES
        elif "prompt" in record and "completion" in record:
            selected = DatasetLayout.PROMPT_COMPLETION
        elif "instruction" in record and ("output" in record or "response" in record):
            selected = DatasetLayout.INSTRUCTION
        elif "text" in record:
            selected = DatasetLayout.TEXT
        else:
            raise DatasetError("cannot infer dataset layout")
    if selected is DatasetLayout.MESSAGES:
        messages = record.get(names.get("messages", "messages"))
        if not isinstance(messages, list) or not messages:
            raise DatasetError("messages records require a non-empty messages list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant", "tool"}
                or not isinstance(message.get("content"), str)
            ):
                raise DatasetError("each message requires a supported role and string content")
        if messages[-1]["role"] != "assistant":
            raise DatasetError("the final supervised message must have role 'assistant'")
        return {"messages": messages}
    if selected is DatasetLayout.INSTRUCTION:
        instruction = _text(record, names.get("instruction", "instruction"))
        extra = record.get(names.get("input", "input"), "")
        if extra and not isinstance(extra, str):
            raise DatasetError("instruction input must be a string")
        prompt = instruction if not extra else f"{instruction}\n\n{extra}"
        response_key = names.get("response", "output" if "output" in record else "response")
        return {"prompt": prompt, "completion": _text(record, response_key)}
    if selected is DatasetLayout.PROMPT_COMPLETION:
        return {
            "prompt": _text(record, names.get("prompt", "prompt")),
            "completion": _text(record, names.get("completion", "completion")),
        }
    if selected is DatasetLayout.TEXT:
        return {"text": _text(record, names.get("text", "text"))}
    raise DatasetError(f"unsupported dataset layout: {selected.value}")


def tokenize_record(
    record: dict[str, Any],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    response_only: bool = True,
) -> PreparedRecord:
    """Tokenize one normalized record and construct exact causal labels."""
    if max_length < 2:
        raise DatasetError("max_length must be at least 2")
    if "messages" in record:
        template = getattr(tokenizer, "apply_chat_template", None)
        if template is None:
            raise DatasetError("messages data requires tokenizer.apply_chat_template")
        messages = record["messages"]
        prompt_ids = list(template(messages[:-1], add_generation_prompt=True, tokenize=True))
        input_ids = list(template(messages, add_generation_prompt=False, tokenize=True))
        prompt_tokens = min(len(prompt_ids), len(input_ids)) if response_only else 0
    elif "prompt" in record:
        prompt_ids = list(tokenizer.encode(record["prompt"], add_special_tokens=True))
        completion_ids = list(tokenizer.encode(record["completion"], add_special_tokens=False))
        input_ids = prompt_ids + completion_ids
        prompt_tokens = len(prompt_ids) if response_only else 0
    else:
        input_ids = list(tokenizer.encode(record["text"], add_special_tokens=True))
        prompt_tokens = 0
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (not input_ids or input_ids[-1] != eos):
        input_ids.append(eos)
    removed = max(len(input_ids) - max_length, 0)
    input_ids = input_ids[:max_length]
    prompt_tokens = min(prompt_tokens, len(input_ids))
    labels = [-100] * prompt_tokens + input_ids[prompt_tokens:]
    if not labels or all(label == -100 for label in labels):
        raise DatasetError("truncation removed every supervised token")
    return PreparedRecord(tuple(input_ids), tuple(labels), prompt_tokens, removed)


def preview_dataset(
    spec: DatasetSpec, tokenizer: Tokenizer, *, max_length: int, limit: int = 3
) -> dict[str, Any]:
    """Return a privacy-safe token and mask summary for the first records."""
    records = load_records(spec)
    prepared = [
        tokenize_record(item, tokenizer, max_length=max_length, response_only=spec.response_only)
        for item in records[:limit]
    ]
    return {
        "dataset_id": spec.dataset_id,
        "record_count": len(records),
        "previewed": len(prepared),
        "sequences": [
            {
                "token_count": len(item.input_ids),
                "supervised_tokens": sum(label != -100 for label in item.labels),
                "prompt_tokens": item.prompt_tokens,
                "truncated_tokens": item.truncated_tokens,
            }
            for item in prepared
        ],
    }


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"record field {key!r} must be a non-empty string")
    return value
