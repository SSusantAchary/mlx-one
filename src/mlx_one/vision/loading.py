"""Strict registry-driven native VLM bundle loading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_one.core.registry import get_registration
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.loading import (
    _prepare_quantized_model,
    _quantization_config,
    _read_json,
    _read_safetensors,
    _resolve,
    _validate_quantized_tensors,
)
from mlx_one.text.tokenizers import HFTokenizerAdapter
from mlx_one.vision.processing import QwenImageProcessor


class VLMModelLoadError(RuntimeError):
    """Raised when a VLM bundle is unsupported, unsafe, or incomplete."""


@dataclass(frozen=True)
class LoadedVLM:
    model: Any
    tokenizer: HFTokenizerAdapter
    processor: QwenImageProcessor
    path: Path
    model_id: str
    revision: str | None
    architecture: str
    quantization: dict[str, Any]
    chat_template: ChatTemplate | None = None


def load_vlm_model(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedVLM:
    try:
        root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
        payload = _read_json(root / "config.json")
        model_type = str(payload.get("model_type", ""))
        registration = get_registration(model_type)
        if registration.modality != "vision-language" or not registration.loader_path:
            raise ValueError(f"model type {model_type!r} has no native VLM task loader")
        config = registration.config_class().from_dict(payload)
        tokenizer = HFTokenizerAdapter.from_directory(root)
        tensors = registration.sanitizer()(_read_safetensors(root), config)
        native = registration.model_class()(config)
        quantization = _quantization_config(payload)
        if quantization:
            _prepare_quantized_model(native, tensors, quantization)
            _validate_quantized_tensors(native, tensors)
        else:
            registration.weight_contract()(config).validate(tensors)
        native.load_weights(list(tensors.items()), strict=True)
        native.eval()
        processor = QwenImageProcessor(tokenizer, config)
        template = ChatTemplate.from_directory(root, model_type, tokenizer)
    except Exception as exc:
        if isinstance(exc, VLMModelLoadError):
            raise
        raise VLMModelLoadError(f"cannot load native VLM: {exc}") from exc
    resolved_revision = revision
    if resolved_revision is None and re.fullmatch(r"[0-9a-f]{40,64}", root.name):
        resolved_revision = root.name
    return LoadedVLM(
        native,
        tokenizer,
        processor,
        root,
        str(model),
        resolved_revision,
        model_type,
        quantization,
        template,
    )
