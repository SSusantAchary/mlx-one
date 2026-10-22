"""Strict registry-driven native VLM bundle loading."""

from __future__ import annotations

import hashlib
import json
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
    processor_identity: str | None = None


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
        model_type = str(payload.get("model_type", "")).replace("-","_")
        registration = get_registration(model_type)
        if registration.modality != "vision-language" or not registration.loader_path:
            raise ValueError(f"model type {model_type!r} has no native VLM task loader")
        if model_type == "lfm2_vl" and payload.get("mlx_one_native_format") != "lfm2-vl-v1":
            raise ValueError("LFM2-VL requires an explicit native-format tensor manifest")
        quantization = _quantization_config(payload)
        config = registration.config_class().from_dict(payload)
        tokenizer = HFTokenizerAdapter.from_directory(root)
        tensors = registration.sanitizer()(_read_safetensors(root), config)
        native = registration.model_class()(config)
        if quantization:
            _prepare_quantized_model(native, tensors, quantization)
            _validate_quantized_tensors(native, tensors)
        else:
            registration.weight_contract()(config).validate(tensors)
        native.load_weights(list(tensors.items()), strict=True)
        native.eval()
        metadata_path = root / "preprocessor_config.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else None
        if model_type != "qwen2_vl" and metadata is None:
            raise ValueError("multimodal loading requires exact processor assets")
        if model_type == "lfm2_vl":
            from mlx_one.vision.lfm_processing import LFMImageProcessor

            processor = LFMImageProcessor(tokenizer,config,metadata)
        else:
            processor = QwenImageProcessor(tokenizer, config,metadata=metadata)
        processor_identity = hashlib.sha256(json.dumps(metadata or {
            "legacy":"qwen2-vl-static-v1"},sort_keys=True).encode()).hexdigest()
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
        processor_identity,
    )
