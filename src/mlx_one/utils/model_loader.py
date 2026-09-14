"""Compatibility helpers routed exclusively through native mlx-one loaders."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from mlx_one.utils.memory import format_memory, get_memory_snapshot

ModelModality = Literal["llm", "vlm", "audio-tts", "audio-stt", "audio-sts", "embedding"]
SUPPORTED_MODALITIES = ("llm", "vlm", "audio-tts", "audio-stt", "audio-sts", "embedding")

_MODALITY_ALIASES = {
    "text": "llm",
    "text-generation": "llm",
    "language": "llm",
    "vision": "vlm",
    "vision-language": "vlm",
    "image-text": "vlm",
    "tts": "audio-tts",
    "text-to-speech": "audio-tts",
    "stt": "audio-stt",
    "speech-to-text": "audio-stt",
    "asr": "audio-stt",
    "sts": "audio-sts",
    "speech-to-speech": "audio-sts",
    "embed": "embedding",
    "embeddings": "embedding",
}


class ModelLoadError(RuntimeError):
    """Raised when an MLX model cannot be loaded cleanly."""


def load_model(
    path_or_repo: str | Path,
    *,
    modality: str = "llm",
    **load_kwargs: Any,
) -> Any:
    """Load an MLX model from a local path or Hugging Face repository ID.

    The historical result shapes are retained where a native task loader exists.
    No third-party MLX ecosystem package is imported as an execution backend.
    """
    model_ref = _normalize_model_ref(path_or_repo)
    resolved_modality = _resolve_modality(modality)
    _validate_local_path(model_ref)
    before = _safe_memory_snapshot()

    try:
        loaded = _load_native(model_ref, resolved_modality, load_kwargs)
    except ModelLoadError:
        raise
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model files were not found for '{model_ref}': {exc}") from exc
    except ValueError as exc:
        message = str(exc)
        if "not supported" in message.lower() or "model type" in message.lower():
            raise ModelLoadError(
                f"Unsupported {resolved_modality} model architecture for '{model_ref}': "
                f"{message}"
            ) from exc
        raise ModelLoadError(f"Unable to load model '{model_ref}': {message}") from exc
    except OSError as exc:
        raise ModelLoadError(f"Unable to access model '{model_ref}': {exc}") from exc
    except Exception as exc:
        raise ModelLoadError(
            f"Unable to load {resolved_modality} model '{model_ref}': {exc}"
        ) from exc

    after = _safe_memory_snapshot()
    _report_memory(model_ref, resolved_modality, before, after)
    return loaded


def load_vlm_model(path_or_repo: str | Path, **load_kwargs: Any) -> Any:
    """Load a vision-language model through the native task loader."""
    return load_model(path_or_repo, modality="vlm", **load_kwargs)


def load_audio_model(
    path_or_repo: str | Path,
    *,
    task: Literal["tts", "stt", "sts"] = "tts",
    **load_kwargs: Any,
) -> Any:
    """Load an audio model natively for an explicitly selected task."""
    return load_model(path_or_repo, modality=f"audio-{task}", **load_kwargs)


def load_embedding_model(path_or_repo: str | Path, **load_kwargs: Any) -> Any:
    """Load an embedding model through native retrieval support."""
    return load_model(path_or_repo, modality="embedding", **load_kwargs)


def _resolve_modality(modality: str) -> ModelModality:
    normalized = modality.strip().lower().replace("_", "-")
    normalized = _MODALITY_ALIASES.get(normalized, normalized)

    if normalized == "audio":
        raise ModelLoadError(
            "Audio modality is ambiguous. Use 'audio-tts', 'audio-stt', or 'audio-sts'."
        )

    if normalized not in SUPPORTED_MODALITIES:
        supported = ", ".join(SUPPORTED_MODALITIES)
        raise ModelLoadError(f"Unsupported model modality '{modality}'. Supported: {supported}.")

    return normalized  # type: ignore[return-value]


def _load_native(model_ref: str, modality: ModelModality, kwargs: dict[str, Any]) -> Any:
    if modality == "llm":
        from mlx_one.text import load_text_model

        bundle = load_text_model(model_ref, **kwargs)
        return bundle.model, bundle.tokenizer
    if modality == "vlm":
        try:
            from mlx_one.vision.loading import load_vlm_model as native_load_vlm
        except ModuleNotFoundError as exc:
            raise ModelLoadError(
                "Native VLM task loading is not available for this build."
            ) from exc
        bundle = native_load_vlm(model_ref, **kwargs)
        return bundle.model, bundle.processor
    if modality == "audio-stt":
        from mlx_one.models.audio.whisper.loading import load_whisper

        return load_whisper(model_ref, **kwargs).model
    if modality in {"audio-tts", "audio-sts"}:
        task = modality.removeprefix("audio-").upper()
        raise ModelLoadError(f"Native {task} model loading is planned but not implemented.")
    from mlx_one.retrieval import load_retrieval_model

    bundle = load_retrieval_model(model_ref, task="embedding", **kwargs)
    return bundle.model, bundle.tokenizer


def _normalize_model_ref(path_or_repo: str | Path) -> str:
    model_ref = str(path_or_repo).strip()
    if not model_ref:
        raise ModelLoadError("Model path or Hugging Face repo ID cannot be empty.")
    return model_ref


def _validate_local_path(model_ref: str) -> None:
    path = Path(model_ref).expanduser()
    looks_like_local = (
        path.is_absolute()
        or model_ref.startswith(".")
        or model_ref.startswith("~")
    )
    if looks_like_local and not path.exists():
        raise ModelLoadError(
            f"Local model path does not exist: {path}. "
            "Pass an existing directory or a Hugging Face repo ID like "
            "'mlx-community/SmolLM-135M-4bit'."
        )
    if path.exists() and not path.is_dir():
        raise ModelLoadError(f"Local model path must be a directory: {path}")


def _safe_memory_snapshot() -> object | None:
    # Importing MLX can abort the interpreter when Metal is unavailable (for
    # example in a sandboxed test runner). Only inspect memory after a real MLX
    # backend has already been imported successfully by the selected loader.
    if "mlx.core" not in sys.modules:
        return None
    try:
        return get_memory_snapshot()
    except BaseException:
        return None


def _report_memory(
    model_ref: str,
    modality: str,
    before: object | None,
    after: object | None,
) -> None:
    if after is None:
        print(f"Loaded {modality} model '{model_ref}'. Memory report unavailable.")
        return

    before_active = getattr(before, "active_bytes", 0) if before is not None else 0
    after_active = getattr(after, "active_bytes", 0)
    delta = max(after_active - before_active, 0)
    print(
        f"Loaded {modality} model '{model_ref}'. "
        f"Memory: {format_memory(after)} (+{delta / 1024**3:.1f} GB)"
    )
