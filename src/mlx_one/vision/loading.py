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
    _check_checkpoint_memory,
    _eos_token_ids,
    _materialize_loaded_model,
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
    eos_token_ids: tuple[int, ...] = ()

    @property
    def context_length(self):
        maximum = self.model.config.text_config.max_position_embeddings
        # Liquid documents this VLM variant at 32K although its text config is larger.
        if self.architecture == "lfm2_vl" and self.model.config.text_config.vocab_size == 128000:
            return min(maximum, 32768)
        return maximum

    @property
    def parameter_count(self):
        return None


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
        model_type = str(payload.get("model_type", "")).replace("-", "_")
        registration = get_registration(model_type)
        if registration.modality != "vision-language" or not registration.loader_path:
            raise ValueError(f"model type {model_type!r} has no native VLM task loader")
        quantization = _quantization_config(payload)
        config = registration.config_class().from_dict(payload)
        tokenizer = HFTokenizerAdapter.from_directory(root)
        _check_checkpoint_memory(root)
        tensors = registration.sanitizer()(_read_safetensors(root), config)
        native = registration.model_class()(config)
        if model_type == "qwen3_5" and any(n.startswith("mtp.") for n in tensors):
            required = {
                n for n in registration.weight_contract()(config).optional if n.startswith("mtp.")
            }
            if required - tensors.keys():
                raise ValueError("incomplete Qwen3.5 MTP tensors")
            native.enable_mtp()
        if model_type in {"qwen2_vl", "qwen2_5_vl", "lfm2_vl"}:
            from mlx_one.models.shared.vlm_weights import quantization_names

            quantization = quantization_names(quantization, model_type)
        if quantization:
            _prepare_quantized_model(native, tensors, quantization)
            _validate_quantized_tensors(native, tensors)
        else:
            registration.weight_contract()(config).validate(tensors)
        native.load_weights(list(tensors.items()), strict=True)
        native.eval()
        _materialize_loaded_model(native)
        metadata = processor_metadata(root)
        if metadata is None:
            raise ValueError("multimodal loading requires exact processor assets")
        processor = make_processor(tokenizer, config, model_type, metadata)
        generation_path = root / "generation_config.json"
        generation = _read_json(generation_path) if generation_path.is_file() else {}
        eos = _eos_token_ids(
            {**payload, **{k: generation[k] for k in ("eos_token_id",) if k in generation}}
        )
        processor_identity = hashlib.sha256(
            json.dumps(metadata or {"legacy": "qwen2-vl-static-v1"}, sort_keys=True).encode()
        ).hexdigest()
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
        eos,
    )


def processor_metadata(root):
    metadata = {}
    for name in ("preprocessor_config.json", "processor_config.json"):
        path = root / name
        if path.is_file():
            value = _read_json(path)
            nested = value.pop("image_processor", {})
            metadata.update(value)
            if not isinstance(nested, dict):
                raise ValueError("image_processor metadata must be an object")
            metadata.update(nested)
    return metadata or None


def make_processor(tokenizer, config, family, metadata):
    if family == "lfm2_vl":
        from mlx_one.vision.lfm_processing import LFMImageProcessor

        return LFMImageProcessor(tokenizer, config, metadata)
    from mlx_one.vision.processing import Qwen25ImageProcessor, Qwen35ImageProcessor

    cls = {
        "qwen2_vl": QwenImageProcessor,
        "qwen2_5_vl": Qwen25ImageProcessor,
        "qwen3_5": Qwen35ImageProcessor,
    }[family]
    return cls(tokenizer, config, metadata=metadata)


def attach_vision_processor(bundle):
    """Attach assets to an already loaded Qwen3.5 bundle without reloading weights."""
    if bundle.architecture != "qwen3_5":
        return bundle
    metadata = processor_metadata(bundle.path)
    if metadata is None:
        return bundle
    processor = make_processor(bundle.tokenizer, bundle.model.config, bundle.architecture, metadata)
    return LoadedVLM(
        bundle.model,
        bundle.tokenizer,
        processor,
        bundle.path,
        bundle.model_id,
        bundle.revision,
        bundle.architecture,
        bundle.quantization,
        bundle.chat_template,
        hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        bundle.eos_token_ids,
    )
