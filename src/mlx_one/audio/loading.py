"""Lazy, task-aware audio loading; unsupported families never fall back."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from mlx_one.core.registry import get_registration

AudioTask = Literal["transcribe", "align", "synthesize", "speech-to-speech"]


class AudioModelLoadError(ValueError):
    """Audio configuration or native task support is unavailable."""


def audio_model_type(config: Mapping[str, Any]) -> str:
    """Resolve the aligner before ASR: both use qwen3_asr at the top level."""

    model_type = config.get("model_type")
    if model_type is None and all(k in config for k in ('encoder', 'joint', 'decoder',
                                                       'model_defaults', 'sample_rate')):
        return 'parakeet_tdt'
    thinker = config.get("thinker_config")
    if model_type == "qwen3_asr" and isinstance(thinker, Mapping):
        if thinker.get("model_type") == "qwen3_forced_aligner":
            return "qwen3_forced_aligner"
    if not isinstance(model_type, str) or not model_type:
        raise AudioModelLoadError("audio config requires a model_type")
    return {"lfm_audio": "lfm2_audio"}.get(model_type, model_type)


def load_audio_model(
    model: str | Path,
    *,
    task: AudioTask = "transcribe",
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load an implemented native audio task from a local or Hub checkpoint.

    Whisper/Distil-Whisper, original Moonshine, Qwen ASR/alignment, and offline
    Parakeet TDT are implemented. Later families are
    eligible only once their registry declares a loader and the requested task.
    Imports and config/task validation do not initialize MLX or execute Hub code.
    """

    task = {"audio.transcribe": "transcribe", "audio.align": "align",
            "audio.generate_speech": "synthesize",
            "audio.speech_to_speech": "speech-to-speech"}.get(task, task)
    if task not in {"transcribe", "align", "synthesize", "speech-to-speech"}:
        raise AudioModelLoadError(f"unsupported audio task: {task}")
    if str(model) == "nvidia/parakeet-tdt-0.6b-v2":
        raise AudioModelLoadError(
            "original Parakeet v2 is a NeMo archive; use the isolated "
            "benchmarks/parakeet_conversion.py tool and load its safetensors directory"
        )
    local = Path(model).expanduser()
    try:
        if local.exists():
            if not local.is_dir():
                raise AudioModelLoadError("local audio model must be a directory")
            config_path = local / "config.json"
        else:
            if isinstance(model, Path) or str(model).startswith((".", "/", "~")):
                raise AudioModelLoadError(f"local audio model directory does not exist: {local}")
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(
                    str(model),
                    "config.json",
                    revision=revision,
                    local_files_only=offline,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            )
            # Resolve weights at this exact commit, even when the input was main.
            revision = config_path.parent.name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise AudioModelLoadError("audio configuration must be a JSON object")
        registration = get_registration(audio_model_type(config))
        if registration.modality not in {"asr", "tts", "audio-language"}:
            raise AudioModelLoadError("checkpoint is not an audio model")
        if not registration.supports(task):
            raise AudioModelLoadError(
                f"native {registration.model_type} does not support task={task!r}"
            )
        return registration.loader()(
            model,
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
        )
    except AudioModelLoadError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise AudioModelLoadError(f"cannot load native audio model: {exc}") from exc
