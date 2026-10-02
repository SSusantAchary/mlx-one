"""Authorized, strict safetensors loading for native SAM3 inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_one.segmentation.processing import Sam3Processor

SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


class SegmentationModelLoadError(RuntimeError):
    """Checkpoint is unsafe, incomplete, or incompatible with the requested task."""


@dataclass(frozen=True)
class LoadedSegmentationModel:
    model: Any
    processor: Sam3Processor
    path: Path
    model_id: str
    revision: str | None
    architecture: str
    task: str

    @property
    def provenance(self):
        return dict(
            model_id=self.model_id,
            revision=self.revision,
            architecture=self.architecture,
            dtype="float32",
            reference="transformers@6133195dcb",
            qualified=False,
        )

    def video_session(self, frames=None, **kwargs):
        from .video import Sam3VideoSession

        return Sam3VideoSession(self, frames=frames, **kwargs)


def load_segmentation_model(model, *, task="concept", revision=None, offline=False, cache_dir=None):
    if task not in ("concept", "interactive", "video"):
        raise ValueError("task must be concept, interactive, or video")
    try:
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        from mlx_one.models.segmentation.sam3 import config as configs
        from mlx_one.models.segmentation.sam3.weights import model_for_config, sanitize_weights
        from mlx_one.text.loading import _read_safetensors

        root = Path(model).expanduser()
        if not root.is_dir():
            if isinstance(model, Path) or str(model).startswith(("/", ".", "~")):
                raise ValueError(f"local checkpoint directory does not exist: {root}")
            root = Path(
                snapshot_download(
                    str(model),
                    revision=revision or (SAM3_REVISION if str(model) == "facebook/sam3" else None),
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_files_only=offline,
                    allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
                )
            )
        payload = json.loads((root / "config.json").read_text())
        model_type = payload.get("model_type")
        if model_type not in ("sam3", "sam3_tracker", "sam3_tracker_video", "sam3_video"):
            raise ValueError(f"unsupported segmentation model type {model_type!r}")
        if payload.get("auto_map") or payload.get("quantization_config"):
            raise ValueError("SAM3 remote code and quantized checkpoints are unsupported")
        target = {"concept": "sam3", "interactive": "sam3_tracker", "video": "sam3_video"}[task]
        if model_type != target and model_type != "sam3_video":
            raise ValueError(f"{model_type} does not provide the {task} workflow")
        selected = (
            payload.get("detector_config", payload)
            if task == "concept"
            else payload.get("tracker_config", payload)
            if task == "interactive"
            else payload
        )
        if task == "interactive":
            selected = {**selected, "model_type": "sam3_tracker"}
        config = {
            "sam3": configs.Sam3Config,
            "sam3_tracker": configs.Sam3TrackerConfig,
            "sam3_video": configs.Sam3VideoConfig,
        }[target].from_dict(selected)
        native = model_for_config(config)
        tensors = sanitize_weights(_read_safetensors(root), config)
        native.load_weights(tensors.items(), strict=True)
        native.eval()
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
        pad = tokenizer_config.get("pad_token", "<|endoftext|>")
        if isinstance(pad, dict):
            pad = pad["content"]
        pad_id = tokenizer.token_to_id(pad)
        if pad_id is None:
            raise ValueError("tokenizer pad token is missing from the vocabulary")
        tokenizer.enable_truncation(max_length=32)
        tokenizer.enable_padding(length=32, pad_id=pad_id, pad_token=pad)
        assets_path = root / "processor_config.json"
        if not assets_path.is_file():
            assets_path = root / "preprocessor_config.json"
        assets = json.loads(assets_path.read_text())
        image_size = (
            config.detector_config.vision_config.image_size
            if task == "video"
            else config.vision_config.image_size
        )
        processor = Sam3Processor(assets, tokenizer, image_size)
        resolved = root.name if re.fullmatch(r"[0-9a-f]{40,64}", root.name) else revision
        return LoadedSegmentationModel(native, processor, root, str(model), resolved, target, task)
    except Exception as exc:
        raise SegmentationModelLoadError(f"cannot load native SAM3 {task}: {exc}") from exc
