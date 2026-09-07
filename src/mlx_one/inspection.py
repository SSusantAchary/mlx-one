"""Metadata-only model inspection for Hugging Face and local directories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelCard, hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from safetensors import SafetensorError, safe_open

from mlx_one.schemas import (
    CapabilityHint,
    CapabilityHintStatus,
    InspectionResult,
    JSONValue,
    Modality,
    ModelDimensions,
    ModelSource,
    ModelSpec,
    Operation,
)

MAX_METADATA_BYTES = 2 * 1024 * 1024

_JSON_FILES = (
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
)
_WEIGHT_FORMATS = {
    ".safetensors": "safetensors",
    ".bin": "pytorch",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".gguf": "gguf",
    ".npz": "npz",
}
_PIPELINE_MODALITIES = {
    "image-text-to-text": Modality.VISION_LANGUAGE,
    "visual-question-answering": Modality.VISION_LANGUAGE,
    "document-question-answering": Modality.VISION_LANGUAGE,
    "automatic-speech-recognition": Modality.ASR,
    "speech-to-text": Modality.ASR,
    "text-to-speech": Modality.TTS,
    "text-to-audio": Modality.TTS,
    "feature-extraction": Modality.EMBEDDING,
    "sentence-similarity": Modality.EMBEDDING,
    "any-to-any": Modality.MULTIMODAL,
    "multimodal": Modality.MULTIMODAL,
}
_TEXT_PIPELINES = {
    "fill-mask",
    "question-answering",
    "summarization",
    "text-classification",
    "text-generation",
    "text2text-generation",
    "token-classification",
    "translation",
    "zero-shot-classification",
}
_ADAPTERS = {
    Modality.TEXT: ("mlx-lm", (Operation.TRAIN, Operation.CONVERT, Operation.EVALUATE)),
    Modality.VISION_LANGUAGE: ("mlx-vlm", (Operation.CONVERT, Operation.EVALUATE)),
    Modality.ASR: ("mlx-audio", (Operation.CONVERT, Operation.EVALUATE)),
    Modality.TTS: ("mlx-audio", (Operation.CONVERT, Operation.EVALUATE)),
    Modality.EMBEDDING: ("mlx-embeddings", (Operation.CONVERT, Operation.EVALUATE)),
}


class InspectionError(RuntimeError):
    """An actionable failure to inspect a model source."""


def inspect_model(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> InspectionResult:
    """Inspect model metadata without loading model weights or importing MLX."""
    requested = str(model)
    path = Path(model).expanduser()
    explicit_local = isinstance(model, Path) or _looks_like_missing_path(requested)
    if path.exists():
        if not path.is_dir():
            raise InspectionError(f"local model path must be a directory: {path}")
        return _inspect_local(path.resolve(), requested_revision=revision)
    if explicit_local:
        raise InspectionError(f"local model directory does not exist: {path}")
    if offline:
        return _inspect_cached_hub(requested, revision=revision, cache_dir=cache_dir)
    return _inspect_hub(requested, revision=revision, cache_dir=cache_dir)


def format_inspection_result(result: InspectionResult) -> str:
    """Format an inspection result for a terminal."""
    model = result.model
    parameters = _format_count(model.parameter_count)
    if model.parameter_count_source:
        parameters = f"{parameters} ({model.parameter_count_source})"
    access = "private" if model.private else "public"
    if model.gated:
        access += ", gated"
    if model.disabled:
        access += ", disabled"
    architectures = ", ".join(model.architectures) or "unknown"
    quantization = json.dumps(model.quantization, sort_keys=True) if model.quantization else "none"
    lines = [
        "mlx-one inspect",
        f"Model: {model.model_id}",
        f"Source: {result.source.value}",
        f"Revision: {model.revision}",
        f"Modality: {model.modality.value}",
        f"Model type: {model.model_type or 'unknown'}",
        f"Architectures: {architectures}",
        f"Parameters: {parameters}",
        f"Weights: {model.weight_format or 'unknown'}, {_format_bytes(model.weight_bytes)}",
        f"Dtype: {model.dtype or 'unknown'}",
        f"Quantization: {quantization}",
        f"Tokenizer: {model.tokenizer_id or 'unknown'}",
        f"Processor: {model.processor_id or 'unknown'}",
        f"License: {model.license or 'unknown'}",
        f"Access: {access}",
        f"Remote code required: {'yes' if model.requires_remote_code else 'no'}",
        "Capability hints:",
    ]
    if result.capabilities:
        for hint in result.capabilities:
            version = f" {hint.installed_version}" if hint.installed_version else ""
            operations = ", ".join(item.value for item in hint.operations) or "unknown"
            lines.append(f"  {hint.adapter}{version}: {hint.status.value} [{operations}]")
            lines.append(f"    {hint.reason}")
    else:
        lines.append("  none")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)


def write_inspection_result(result: InspectionResult, output: str | Path) -> Path:
    """Atomically write an inspection result as JSON."""
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(f"{result.to_json()}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise InspectionError(f"could not write inspection result to {path}: {exc}") from exc
    return path


def _inspect_hub(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
) -> InspectionResult:
    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except GatedRepoError as exc:
        raise InspectionError(
            f"access to {repo_id!r} is gated; authenticate with `hf auth login` or HF_TOKEN"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise InspectionError(
            f"Hugging Face model {repo_id!r} was not found or is private; "
            "check the ID and authentication"
        ) from exc
    except RevisionNotFoundError as exc:
        raise InspectionError(f"revision {revision!r} was not found for {repo_id!r}") from exc
    except HfHubHTTPError as exc:
        raise InspectionError(f"Hugging Face inspection failed for {repo_id!r}: {exc}") from exc
    except HFValidationError as exc:
        raise InspectionError(f"invalid Hugging Face model ID {repo_id!r}: {exc}") from exc
    except (OSError, TimeoutError) as exc:
        raise InspectionError(f"could not reach Hugging Face for {repo_id!r}: {exc}") from exc

    siblings = {item.rfilename: item for item in info.siblings or ()}
    config = _mapping_or_empty(info.config)
    supplemental: dict[str, dict[str, Any]] = {}
    if not config and "config.json" in siblings:
        config = _load_hub_json(
            repo_id,
            "config.json",
            revision=info.sha or revision,
            cache_dir=cache_dir,
            size=_sibling_size(siblings["config.json"]),
        ) or {}
    for filename in _JSON_FILES[1:]:
        if filename in siblings:
            loaded = _load_hub_json(
                repo_id,
                filename,
                revision=info.sha or revision,
                cache_dir=cache_dir,
                size=_sibling_size(siblings[filename]),
            )
            if loaded is not None:
                supplemental[filename] = loaded

    card = info.card_data
    pipeline_tag = info.pipeline_tag or _card_value(card, "pipeline_tag")
    library_name = info.library_name or _card_value(card, "library_name")
    base_models = _as_strings(info.base_models or _card_value(card, "base_model"))
    license_name = _string_or_none(_card_value(card, "license"))
    tags = tuple(str(tag) for tag in info.tags or ())
    formats, weight_bytes = _hub_weights(siblings)
    parameter_count, parameter_source, dtype = _hub_parameters(info, config)
    model_id = info.id or repo_id
    resolved_revision = info.sha or revision or "unresolved"
    model_spec = _model_spec(
        model_id=model_id,
        revision=resolved_revision,
        config=config,
        supplemental=supplemental,
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        base_models=base_models,
        license_name=license_name,
        tags=tags,
        formats=formats,
        weight_bytes=weight_bytes,
        parameter_count=parameter_count,
        parameter_source=parameter_source,
        dtype=dtype,
        gated=bool(info.gated),
        private=bool(info.private),
        disabled=bool(info.disabled),
        metadata={"used_storage_bytes": info.used_storage},
        file_names=tuple(siblings),
    )
    warnings = _warnings(model_spec, resolved=info.sha is not None)
    return InspectionResult(
        model=model_spec,
        source=ModelSource.HUGGING_FACE,
        requested_revision=revision,
        capabilities=_capability_hints(model_spec.modality),
        warnings=tuple(warnings),
    )


def _inspect_cached_hub(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
) -> InspectionResult:
    try:
        cached = snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError) as exc:
        raise InspectionError(
            f"no cached metadata is available for {repo_id!r}; rerun without --offline first"
        ) from exc
    except (RepositoryNotFoundError, RevisionNotFoundError, OSError) as exc:
        raise InspectionError(f"could not inspect cached model {repo_id!r}: {exc}") from exc

    path = Path(cached)
    result = _inspect_local(
        path,
        requested_revision=revision,
        model_id=repo_id,
        source=ModelSource.HUGGING_FACE,
        revision_override=_cached_revision(path, revision),
    )
    return InspectionResult(
        model=result.model,
        source=result.source,
        requested_revision=revision,
        capabilities=result.capabilities,
        warnings=("Inspected an existing Hugging Face cache snapshot in offline mode.",)
        + result.warnings,
    )


def _inspect_local(
    path: Path,
    *,
    requested_revision: str | None,
    model_id: str | None = None,
    source: ModelSource = ModelSource.LOCAL,
    revision_override: str | None = None,
) -> InspectionResult:
    files = tuple(item for item in path.rglob("*") if item.is_file())
    recognized = [
        item
        for item in files
        if item.name in _JSON_FILES
        or item.name.lower() == "readme.md"
        or _weight_format(item) is not None
    ]
    if not recognized:
        raise InspectionError(f"no recognized model metadata or weights found in {path}")

    metadata_files: dict[str, dict[str, Any]] = {}
    for filename in _JSON_FILES:
        candidate = path / filename
        if candidate.exists():
            metadata_files[filename] = _load_local_json(candidate)
    config = metadata_files.get("config.json", {})
    card_data = _load_local_card(path / "README.md")
    pipeline_tag = _string_or_none(card_data.get("pipeline_tag"))
    library_name = _string_or_none(card_data.get("library_name"))
    base_models = _as_strings(card_data.get("base_model"))
    license_name = _string_or_none(card_data.get("license") or config.get("license"))
    tags = _as_strings(card_data.get("tags"))
    weight_files = tuple(item for item in files if _weight_format(item) is not None)
    formats = tuple(sorted({_weight_format(item) for item in weight_files if _weight_format(item)}))
    weight_bytes = sum(item.stat().st_size for item in weight_files)
    parameter_count, parameter_source, dtype, header_hashes = _local_parameters(weight_files)
    if parameter_count is None:
        parameter_count = _declared_parameters(config)
        parameter_source = "declared" if parameter_count is not None else "unknown"
    revision = revision_override or (
        f"local:{_local_manifest_digest(path, recognized, header_hashes)}"
    )
    supplemental = {key: value for key, value in metadata_files.items() if key != "config.json"}
    model_spec = _model_spec(
        model_id=model_id or str(path),
        revision=revision,
        config=config,
        supplemental=supplemental,
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        base_models=base_models,
        license_name=license_name,
        tags=tags,
        formats=formats,
        weight_bytes=weight_bytes or None,
        parameter_count=parameter_count,
        parameter_source=parameter_source,
        dtype=dtype,
        gated=False,
        private=False,
        disabled=False,
        metadata={
            "revision_source": (
                "hugging-face-cache" if revision_override else "local-metadata-manifest"
            )
        },
        file_names=tuple(str(item.relative_to(path)) for item in files),
    )
    warnings = _warnings(model_spec, resolved=True)
    if requested_revision is not None and source is ModelSource.LOCAL:
        warnings.append("The revision option is ignored for a local model directory.")
    return InspectionResult(
        model=model_spec,
        source=source,
        requested_revision=requested_revision,
        capabilities=_capability_hints(model_spec.modality),
        warnings=tuple(warnings),
    )


def _model_spec(
    *,
    model_id: str,
    revision: str,
    config: Mapping[str, Any],
    supplemental: Mapping[str, Mapping[str, Any]],
    pipeline_tag: str | None,
    library_name: str | None,
    base_models: tuple[str, ...],
    license_name: str | None,
    tags: tuple[str, ...],
    formats: tuple[str, ...],
    weight_bytes: int | None,
    parameter_count: int | None,
    parameter_source: str,
    dtype: str | None,
    gated: bool,
    private: bool,
    disabled: bool,
    metadata: Mapping[str, JSONValue],
    file_names: tuple[str, ...],
) -> ModelSpec:
    modality = _infer_modality(pipeline_tag, config, tags, library_name)
    tokenizer_present = bool(
        config.get("tokenizer_class")
        or "tokenizer_config.json" in supplemental
        or any(
            Path(name).name
            in {
                "tokenizer_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "vocab.json",
                "vocab.txt",
            }
            for name in file_names
        )
    )
    processor_present = bool(
        config.get("processor_class")
        or config.get("vision_config")
        or config.get("audio_config")
        or any(
            name in supplemental
            for name in ("processor_config.json", "preprocessor_config.json")
        )
        or any(
            Path(name).name in {"processor_config.json", "preprocessor_config.json"}
            for name in file_names
        )
    )
    combined_metadata: dict[str, JSONValue] = dict(metadata)
    combined_metadata["tags"] = list(tags)
    return ModelSpec(
        model_id=model_id,
        revision=revision,
        modality=modality,
        architectures=_architectures(config),
        model_type=_string_or_none(config.get("model_type")),
        base_models=base_models,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        parameter_count=parameter_count,
        parameter_count_source=parameter_source,
        weight_format=_single_or_mixed(formats),
        weight_bytes=weight_bytes,
        dtype=dtype,
        quantization=_quantization(config),
        tokenizer_id=model_id if tokenizer_present else None,
        processor_id=model_id if processor_present else None,
        license=license_name,
        gated=gated,
        private=private,
        disabled=disabled,
        requires_remote_code=_requires_remote_code(config, supplemental),
        dimensions=_model_dimensions(config),
        metadata=combined_metadata,
    )


def _load_hub_json(
    repo_id: str,
    filename: str,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    size: int | None,
) -> dict[str, Any] | None:
    if size is None or size > MAX_METADATA_BYTES:
        return None
    try:
        path = hf_hub_download(
            repo_id,
            filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    except (HfHubHTTPError, OSError):
        return None
    return _load_local_json(Path(path))


def _load_local_json(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size > MAX_METADATA_BYTES:
            raise InspectionError(
                f"metadata file {path} is larger than {MAX_METADATA_BYTES} bytes"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InspectionError(f"invalid metadata file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InspectionError(f"metadata file {path} must contain a JSON object")
    return payload


def _load_local_card(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            raise InspectionError(
                f"metadata file {path} is larger than {MAX_METADATA_BYTES} bytes"
            )
        card = ModelCard.load(str(path))
        return dict(card.data.to_dict())
    except InspectionError:
        raise
    except Exception as exc:  # ModelCard raises multiple parser-specific exceptions
        raise InspectionError(f"invalid model card {path}: {exc}") from exc


def _local_parameters(
    weight_files: tuple[Path, ...],
) -> tuple[int | None, str, str | None, dict[str, str]]:
    safetensors_files = tuple(
        path for path in weight_files if _weight_format(path) == "safetensors"
    )
    if not safetensors_files:
        return None, "unknown", None, {}
    total = 0
    dtypes: set[str] = set()
    header_hashes: dict[str, str] = {}
    for path in safetensors_files:
        try:
            with path.open("rb") as file:
                header_size_bytes = file.read(8)
                if len(header_size_bytes) != 8:
                    raise ValueError("file is too small")
                header_size = int.from_bytes(header_size_bytes, "little")
                if header_size > MAX_METADATA_BYTES:
                    raise ValueError("safetensors header is too large")
                header = file.read(header_size)
                if len(header) != header_size:
                    raise ValueError("incomplete safetensors header")
                header_hashes[str(path)] = hashlib.sha256(header).hexdigest()
            with safe_open(path, framework="numpy") as handle:
                for key in handle.keys():
                    tensor_slice = handle.get_slice(key)
                    shape = tensor_slice.get_shape()
                    count = 1
                    for dimension in shape:
                        count *= dimension
                    total += count
                    dtypes.add(str(tensor_slice.get_dtype()))
        except (OSError, ValueError, SafetensorError) as exc:
            raise InspectionError(f"invalid safetensors metadata in {path}: {exc}") from exc
    return total, "local-safetensors-header", _single_or_mixed(tuple(sorted(dtypes))), header_hashes


def _hub_parameters(info: Any, config: Mapping[str, Any]) -> tuple[int | None, str, str | None]:
    safetensors = info.safetensors
    if safetensors is not None and isinstance(safetensors.total, int):
        dtypes = tuple(
            sorted(str(dtype) for dtype, count in safetensors.parameters.items() if count)
        )
        return safetensors.total, "hub-safetensors", _single_or_mixed(dtypes)
    declared = _declared_parameters(config)
    return declared, "declared" if declared is not None else "unknown", None


def _declared_parameters(config: Mapping[str, Any]) -> int | None:
    for key in ("num_parameters", "parameter_count", "n_params"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _hub_weights(siblings: Mapping[str, Any]) -> tuple[tuple[str, ...], int | None]:
    formats: set[str] = set()
    total = 0
    known_size = False
    for filename, sibling in siblings.items():
        weight_format = _weight_format(Path(filename))
        if weight_format is None:
            continue
        formats.add(weight_format)
        size = _sibling_size(sibling)
        if size is not None:
            total += size
            known_size = True
    return tuple(sorted(formats)), total if known_size else None


def _capability_hints(modality: Modality) -> tuple[CapabilityHint, ...]:
    if modality is Modality.UNKNOWN or modality is Modality.MULTIMODAL:
        return (
            CapabilityHint(
                adapter="unknown",
                status=CapabilityHintStatus.UNKNOWN,
                reason="The modality does not map to one first-party MLX adapter.",
            ),
        )
    adapter = _ADAPTERS.get(modality)
    if adapter is None:
        return ()
    distribution, operations = adapter
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return (
            CapabilityHint(
                adapter=distribution,
                status=CapabilityHintStatus.BACKEND_UNAVAILABLE,
                operations=operations,
                reason=f"Install the optional {distribution} backend to evaluate this candidate.",
            ),
        )
    return (
        CapabilityHint(
            adapter=distribution,
            installed_version=version,
            status=CapabilityHintStatus.CANDIDATE,
            operations=operations,
            reason=(
                "Backend availability is a hint; architecture compatibility is not yet verified."
            ),
        ),
    )


def _infer_modality(
    pipeline_tag: str | None,
    config: Mapping[str, Any],
    tags: tuple[str, ...],
    library_name: str | None,
) -> Modality:
    normalized_pipeline = (pipeline_tag or "").lower()
    if normalized_pipeline in _PIPELINE_MODALITIES:
        return _PIPELINE_MODALITIES[normalized_pipeline]
    if normalized_pipeline in _TEXT_PIPELINES:
        return Modality.TEXT

    names = " ".join(
        [
            str(config.get("model_type", "")),
            *[str(item) for item in config.get("architectures", ())],
        ]
    ).lower()
    if config.get("vision_config") is not None or any(
        marker in names
        for marker in (
            "vision2seq",
            "visionlanguage",
            "qwen2vl",
            "qwen3vl",
            "llava",
            "pixtral",
            "paligemma",
            "idefics",
        )
    ):
        return Modality.VISION_LANGUAGE
    if any(marker in names for marker in ("texttospeech", "speecht5", "bark", "tts")):
        return Modality.TTS
    if config.get("audio_config") is not None or any(
        marker in names for marker in ("whisper", "ctc", "speechrecognition", "wav2vec")
    ):
        return Modality.ASR
    if any(marker in names for marker in ("sentence", "embedding", "clipmodel")):
        return Modality.EMBEDDING
    if names.strip():
        return Modality.TEXT

    fallback = " ".join((*tags, library_name or "")).lower()
    for task, modality in _PIPELINE_MODALITIES.items():
        if task in fallback:
            return modality
    if any(task in fallback for task in _TEXT_PIPELINES):
        return Modality.TEXT
    return Modality.UNKNOWN


def _requires_remote_code(
    config: Mapping[str, Any], supplemental: Mapping[str, Mapping[str, Any]]
) -> bool:
    values: list[Mapping[str, Any]] = [config, *supplemental.values()]
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        values.append(text_config)
    return any(bool(value.get("auto_map")) for value in values)


def _quantization(config: Mapping[str, Any]) -> dict[str, JSONValue]:
    value = config.get("quantization_config", config.get("quantization", {}))
    if not isinstance(value, Mapping):
        return {}
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return {}
    return dict(value)


def _architectures(config: Mapping[str, Any]) -> tuple[str, ...]:
    return _as_strings(config.get("architectures"))


def _model_dimensions(config: Mapping[str, Any]) -> ModelDimensions:
    """Extract common transformer dimensions without importing model code."""
    text_config = config.get("text_config")
    source = text_config if isinstance(text_config, Mapping) else config

    def positive_int(*names: str) -> int | None:
        for name in names:
            value = source.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    hidden_size = positive_int("hidden_size", "n_embd", "d_model")
    attention_heads = positive_int("num_attention_heads", "n_head", "attention_heads")
    head_dimension = positive_int("head_dim")
    if head_dimension is None and hidden_size is not None and attention_heads:
        head_dimension = hidden_size // attention_heads
    return ModelDimensions(
        hidden_size=hidden_size,
        layer_count=positive_int("num_hidden_layers", "n_layer", "num_layers"),
        attention_heads=attention_heads,
        key_value_heads=positive_int("num_key_value_heads", "n_kv_heads") or attention_heads,
        head_dimension=head_dimension,
        intermediate_size=positive_int("intermediate_size", "n_inner", "ffn_dim"),
        vocabulary_size=positive_int("vocab_size", "vocabulary_size"),
    )


def _warnings(model: ModelSpec, *, resolved: bool) -> list[str]:
    warnings = []
    if model.parameter_count is None:
        warnings.append("Parameter count is unavailable from metadata.")
    if model.license is None:
        warnings.append("License metadata is unavailable; review the source before use.")
    if not model.architectures:
        warnings.append("Architecture metadata is unavailable.")
    if model.model_type is None:
        warnings.append("Model type is unavailable from metadata.")
    if model.modality is Modality.UNKNOWN:
        warnings.append("Modality could not be determined from metadata.")
    if model.weight_format is None:
        warnings.append("Weight format is unavailable from metadata.")
    if model.dtype is None and model.weight_format == "safetensors":
        warnings.append("Safetensors dtype metadata is unavailable.")
    if model.modality in (Modality.TEXT, Modality.EMBEDDING) and model.tokenizer_id is None:
        warnings.append("Tokenizer metadata is unavailable.")
    if model.modality in (Modality.VISION_LANGUAGE, Modality.ASR, Modality.TTS):
        if model.processor_id is None:
            warnings.append("Processor metadata is unavailable.")
    if not resolved:
        warnings.append("The requested Hub revision could not be resolved to an exact commit SHA.")
    return warnings


def _local_manifest_digest(
    root: Path, recognized: list[Path], header_hashes: Mapping[str, str]
) -> str:
    entries = []
    for path in sorted(recognized):
        entry: dict[str, JSONValue] = {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
        }
        if path.name in _JSON_FILES or path.name.lower() == "readme.md":
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif str(path) in header_hashes:
            entry["header_sha256"] = header_hashes[str(path)]
        entries.append(entry)
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _cached_revision(path: Path, requested: str | None) -> str:
    if path.parent.name == "snapshots" and len(path.name) >= 7:
        return path.name
    return requested or "cached-unresolved"


def _weight_format(path: Path) -> str | None:
    return _WEIGHT_FORMATS.get(path.suffix.lower())


def _sibling_size(sibling: Any) -> int | None:
    size = getattr(sibling, "size", None)
    if isinstance(size, int):
        return size
    lfs = getattr(sibling, "lfs", None)
    lfs_size = getattr(lfs, "size", None)
    return lfs_size if isinstance(lfs_size, int) else None


def _card_value(card: Any, name: str) -> Any:
    return getattr(card, name, None) if card is not None else None


def _mapping_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _single_or_mixed(values: tuple[str, ...]) -> str | None:
    unique = tuple(dict.fromkeys(value for value in values if value))
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else "mixed"


def _looks_like_missing_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or "\\" in value


def _format_count(value: int | None) -> str:
    if value is None:
        return "unknown"
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return str(value)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "size unknown"
    for divisor, suffix in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if value >= divisor:
            return f"{value / divisor:.2f} {suffix}"
    return f"{value} bytes"
