"""Strict native OCR loading, reusing safetensors and affine quantization support."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from mlx_one.ocr.registry import capabilities, ocr_family


@dataclass(frozen=True)
class LoadedOCR:
    model: object
    tokenizer: object
    config: object
    path: Path
    model_id: str
    revision: str | None
    architecture: str
    quantization: dict
    processor_metadata: dict
    capabilities: object
    eos_token_ids: tuple[int, ...]
    context_length: int

    @property
    def resident_bytes(self):
        from mlx_one.engine.cache import array_nbytes

        return array_nbytes(self.model.parameters())


def make_config(payload, family):
    module = importlib.import_module(f"mlx_one.models.ocr.{family}.config")
    cls = module.TrOCRConfig if family == "trocr" else module.ModelConfig
    config = cls.from_dict(json.loads(json.dumps(payload)))
    # Ported reference configs separate parsing and construction of nested components.
    for name in ("vision_config", "text_config", "projector_config"):
        item = getattr(config, name, None)
        nested_cls = getattr(
            module,
            {
                "vision_config": "VisionConfig",
                "text_config": "TextConfig",
                "projector_config": "ProjectorConfig",
            }[name],
            None,
        )
        if isinstance(item, dict) and nested_cls is not None:
            setattr(config, name, nested_cls.from_dict(item))
    if family == "florence2":
        for name in ("image_feature_source", "image_pos_embed", "visual_temporal_embedding"):
            if name in payload.get("vision_config", {}):
                setattr(config, name, payload["vision_config"][name])
    if family == "paddleocr_vl":
        config.vision_config.output_dim = config.text_config.hidden_size
    if family == "deepseek_ocr":
        if payload.get("use_mla") or config.text_config.attn_type != "LlamaAttention":
            raise ValueError("only the pinned DeepSeek-OCR non-MLA decoder is supported")
    return config


def make_model(config, family):
    module = importlib.import_module(f"mlx_one.models.ocr.{family}.model")
    return (module.TrOCR if family == "trocr" else module.Model)(config)


def load_ocr_model(model, *, revision=None, offline=True, cache_dir=None, handwriting=False):
    from mlx.utils import tree_flatten

    from mlx_one.ocr.tokenizers import load_tokenizer
    from mlx_one.text.loading import (
        _check_checkpoint_memory,
        _materialize_loaded_model,
        _prepare_quantized_model,
        _quantization_config,
        _read_safetensors,
        _resolve,
        _validate_quantized_tensors,
    )

    root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
    payload = json.loads((root / "config.json").read_text())
    family = ocr_family(payload)
    config = make_config(payload, family)
    tokenizer = load_tokenizer(root, family)
    metadata = {}
    for name in ("preprocessor_config.json", "processor_config.json"):
        if (root / name).is_file():
            metadata.update(json.loads((root / name).read_text()))
    if family != "got_ocr2" and not metadata:
        raise ValueError("OCR requires the checkpoint processor configuration")
    _check_checkpoint_memory(root)
    native = make_model(config, family)
    tensors = native.sanitize(_read_safetensors(root))
    # All mapping is checked against the actual target operation's shape. Already
    # converted MLX convolutions remain unchanged; only verified OIHW becomes OHWI.
    expected = dict(tree_flatten(native.parameters()))
    for name, tensor in list(tensors.items()):
        target = expected.get(name)
        if tensor.ndim == 4 and target is not None and tensor.shape != target.shape:
            converted = tensor.transpose(0, 2, 3, 1)
            if converted.shape == target.shape:
                tensors[name] = converted
    if family == "deepseek_ocr":
        tensors = native.language_model.sanitize(tensors)
    quant = _quantization_config(payload)
    if quant:
        if quant.get("mode", "affine") != "affine":
            raise ValueError("OCR loader supports affine Q4/Q8, not implicit alternate formats")
        _prepare_quantized_model(native, tensors, quant)
        _validate_quantized_tensors(native, tensors)
    native.load_weights(list(tensors.items()), strict=True)
    native.eval()
    _materialize_loaded_model(native)
    cfg = getattr(config, "text_config", getattr(config, "decoder", config))
    context = getattr(cfg, "max_position_embeddings", 8192)
    eos = payload.get("eos_token_id", getattr(cfg, "eos_token_id", None))
    eos = (eos,) if isinstance(eos, int) else tuple(eos or ())
    if family == "got_ocr2":
        eos = tuple(sorted(set(eos) | {151643, 151645}))
    resolved = revision
    cached_config = root / ".cache/huggingface/download/config.json.metadata"
    if cached_config.is_file():
        actual_revision = cached_config.read_text().splitlines()[0]
        if revision is not None and actual_revision != revision:
            raise ValueError("requested revision does not match downloaded checkpoint metadata")
        resolved = actual_revision
    if (
        resolved is None
        and len(root.name) == 40
        and all(c in "0123456789abcdef" for c in root.name)
    ):
        resolved = root.name
    # Handwriting is never inferred merely from the shared TrOCR architecture.
    handwriting = (
        handwriting
        or str(model) == "microsoft/trocr-base-handwritten"
        or (resolved == "eaacaf452b06415df8f10bb6fad3a4c11e609406")
    )
    return LoadedOCR(
        native,
        tokenizer,
        config,
        root,
        str(model),
        resolved,
        family,
        quant,
        metadata,
        capabilities(family, handwriting=handwriting),
        eos,
        context,
    )
