"""Safe registry-driven native causal-language model loading."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from mlx_one.core.registry import get_registration
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.tokenizer import GPT2Tokenizer
from mlx_one.text.tokenizers import HFTokenizerAdapter, TextTokenizer


class TextModelLoadError(RuntimeError):
    """Raised when a native text bundle is unsafe, incomplete, or unsupported."""


@dataclass(frozen=True)
class LoadedTextModel:
    model: Any
    tokenizer: TextTokenizer
    path: Path
    model_id: str
    revision: str | None
    architecture: str = "gpt2"
    context_length: int = 0
    quantization: dict[str, Any] = field(default_factory=dict)
    chat_template: ChatTemplate | None = None
    parameter_count: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    tokenizer_identity: str | None = None
    adapter_identity: str | None = None
    weight_hashes: dict[str, str] = field(default_factory=dict)


def resolve_text_model_type(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> str:
    local = Path(model).expanduser()
    if not local.exists() and (
        isinstance(model, Path) or str(model).startswith((".", "/", "~"))
    ):
        raise TextModelLoadError(f"local text model directory does not exist: {local}")
    try:
        config_path = (
            local / "config.json"
            if local.is_dir()
            else Path(
                hf_hub_download(
                    str(model),
                    "config.json",
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    local_files_only=offline,
                )
            )
        )
        return str(_read_json(config_path).get("model_type", ""))
    except Exception as exc:
        if isinstance(exc, TextModelLoadError):
            raise
        raise TextModelLoadError(f"cannot resolve text model type: {exc}") from exc


def load_text_model(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
    adapter_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    expected_files: dict[str, str] | None = None,
    _qwen3_candidate: bool = False,
) -> LoadedTextModel:
    root = _resolve(checkpoint_path or model, revision=revision,
                    offline=offline, cache_dir=cache_dir)
    from mlx_one.conversion import ConversionError, verify_conversion_manifest

    try:
        converted = verify_conversion_manifest(root)
    except (ConversionError, ValueError, OSError) as exc:
        raise TextModelLoadError(f"invalid converted checkpoint: {exc}") from exc
    if converted is not None and revision not in {None, converted["artifact_digest"]}:
        raise TextModelLoadError("local conversion revision differs from artifact digest")
    identity, resolved_revision, weight_hashes = checkpoint_identity(
        root, model, revision=revision, converted=converted,
        explicit_path=checkpoint_path is not None, expected_files=expected_files)
    config_data = _read_json(root / "config.json")
    model_type = str(config_data.get("model_type", ""))
    if _qwen3_candidate and model_type != "qwen3":
        raise TextModelLoadError("the upstream candidate loader requires model_type=qwen3")
    try:
        registration = get_registration(model_type)
        if not registration.loader_path or not registration.supports("text-generation"):
            raise ValueError(f"model type {model_type!r} has no native text loader")
        quantization = _quantization_config(config_data)
        config = registration.config_class().from_dict(config_data)
        tokenizer_root = (
            _resolve_tokenizer(
                tokenizer_source, revision=revision, offline=offline, cache_dir=cache_dir
            )
            if tokenizer_source is not None
            else root
        )
        tokenizer = _load_tokenizer(tokenizer_root, model_type)
        chat_template = ChatTemplate.from_directory(tokenizer_root, model_type, tokenizer)
        _check_checkpoint_memory(root)
        tensors = registration.sanitizer()(_read_safetensors(root), config)
        if _qwen3_candidate:
            from mlx_one.models.language.qwen3.upstream_model import Qwen3UpstreamForCausalLM

            native = Qwen3UpstreamForCausalLM(config)
        else:
            native = registration.model_class()(config)
        if model_type == "qwen3_5" and any(name.startswith("mtp.") for name in tensors):
            required_mtp = {
                name
                for name in registration.weight_contract()(config).optional
                if name.startswith("mtp.")
            }
            missing_mtp = sorted(required_mtp - set(tensors))
            if missing_mtp:
                raise ValueError(f"incomplete Qwen3.5 MTP tensors: {', '.join(missing_mtp)}")
            native.enable_mtp()
        generation_path = root / "generation_config.json"
        generation_config = _read_json(generation_path) if generation_path.is_file() else {}
        eos_token_ids = _eos_token_ids({**config_data, **{
            key: generation_config[key] for key in ("eos_token_id",) if key in generation_config
        }})
        if quantization:
            _prepare_quantized_model(native, tensors, quantization)
            _validate_quantized_tensors(native, tensors)
        else:
            registration.weight_contract()(config).validate(tensors)
        native.load_weights(list(tensors.items()), strict=True)
        if adapter_path is not None:
            from mlx_one.tuning import load_adapter

            load_adapter(
                native,
                adapter_path,
                base_model_id=str(model),
                base_revision=revision,
            )
        native.eval()
        _materialize_loaded_model(native)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if isinstance(exc, TextModelLoadError):
            raise
        raise TextModelLoadError(f"cannot load native {model_type or 'text'} model: {exc}") from exc
    return LoadedTextModel(
        native,
        tokenizer,
        root,
        identity,
        resolved_revision,
        architecture=model_type,
        context_length=_context_length(config),
        quantization=quantization,
        chat_template=chat_template,
        parameter_count=_parameter_count(config_data),
        eos_token_ids=eos_token_ids,
        weight_hashes=weight_hashes,
        tokenizer_identity=_asset_identity(tokenizer_root),
        adapter_identity=(
            _asset_identity(Path(adapter_path).expanduser())
            if adapter_path is not None
            else None
        ),
    )


def checkpoint_identity(root, requested, *, revision=None, converted=None,
                        explicit_path=False, expected_files=None):
    """Keep repository identity, snapshot location, revision and content proof distinct.

    A local Hub snapshot carries its repository/revision in the cache layout. An
    arbitrary local copy needs an exact content proof before binding a canonical ID.
    No basename or HTTP alias is accepted as a weight verification mechanism.
    """
    root = Path(root).resolve(strict=True)
    canonical = converted["source"] if converted else None
    resolved_revision = converted["artifact_digest"] if converted else revision
    requested_is_path = Path(requested).expanduser().is_dir()
    if (converted is None and root.parent.name == "snapshots"
            and root.parent.parent.name.startswith("models--")):
        canonical = root.parent.parent.name.removeprefix("models--").replace("--", "/")
        if (revision is not None and revision != root.name and
                (explicit_path or requested_is_path or re.fullmatch(r"[0-9a-f]{40,64}", revision))):
            raise TextModelLoadError("checkpoint snapshot revision mismatch")
        resolved_revision = root.name
    elif resolved_revision is None and re.fullmatch(r"[0-9a-f]{40,64}", root.name):
        resolved_revision = root.name
    if canonical and not requested_is_path and str(requested) != canonical:
        raise TextModelLoadError("canonical model ID does not match checkpoint origin")
    if explicit_path and canonical is None and not expected_files:
        raise TextModelLoadError(
            "binding a local checkpoint to a canonical ID requires file hashes")
    hashes = {}
    if expected_files is not None:
        from mlx_one.runtime_baseline import file_digest

        actual = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*"))
                  if p.is_file() and not any(part.startswith(".")
                                            for part in p.relative_to(root).parts)}
        if actual != expected_files or not any(name.endswith(".safetensors") for name in actual):
            raise TextModelLoadError("checkpoint file hash verification failed")
        hashes = {name: digest for name, digest in actual.items() if name.endswith(".safetensors")}
    return canonical or str(requested), resolved_revision, hashes


def _asset_identity(root: Path) -> str:
    """Create a process-safe identity without hashing multi-gigabyte weight contents."""

    candidates = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "adapter_config.json",
        "adapter_model.safetensors",
    )
    records = []
    for name in candidates:
        path = root / name if root.is_dir() else root
        if not path.is_file():
            continue
        stat = path.stat()
        digest = None
        if stat.st_size <= 16 * 1024**2:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append((path.name, stat.st_size, stat.st_mtime_ns, digest))
        if not root.is_dir():
            break
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _materialize_loaded_model(model):
    """Finish checkpoint IO on the loading thread before handing arrays to Engine.

    Deferred MLX file reads/casts can retain streams belonging to their creator
    thread. Fully loading parameters also makes load-time failures deterministic.
    """
    import mlx.core as mx

    mx.eval(model.parameters())


def _load_tokenizer(root: Path, model_type: str) -> TextTokenizer:
    try:
        if (
            model_type == "gpt2"
            and (root / "vocab.json").is_file()
            and (root / "merges.txt").is_file()
        ):
            return GPT2Tokenizer.from_directory(root)
        return HFTokenizerAdapter.from_directory(root)
    except ValueError as exc:
        raise TextModelLoadError(f"invalid tokenizer assets: {exc}") from exc


def _resolve(
    model: str | Path,
    *,
    revision: str | None,
    offline: bool,
    cache_dir: str | Path | None,
) -> Path:
    local = Path(model).expanduser()
    if local.exists():
        if not local.is_dir():
            raise TextModelLoadError("local text model must be a directory")
        return local.resolve()
    if isinstance(model, Path) or str(model).startswith((".", "/", "~")):
        raise TextModelLoadError(f"local text model directory does not exist: {local}")
    try:
        return Path(
            snapshot_download(
                str(model),
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=offline,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                    "vocab.json",
                    "merges.txt",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
            )
        )
    except Exception as exc:
        raise TextModelLoadError(f"cannot resolve native text model: {exc}") from exc


def _resolve_tokenizer(
    source: str | Path,
    *,
    revision: str | None,
    offline: bool,
    cache_dir: str | Path | None,
) -> Path:
    local = Path(source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise TextModelLoadError("tokenizer source must be a directory")
        return local.resolve()
    if isinstance(source, Path) or str(source).startswith((".", "/", "~")):
        raise TextModelLoadError(f"local tokenizer directory does not exist: {local}")
    try:
        return Path(
            snapshot_download(
                str(source),
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=offline,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                    "vocab.json",
                    "merges.txt",
                ],
            )
        )
    except Exception as exc:
        raise TextModelLoadError(f"cannot resolve tokenizer: {exc}") from exc


def _quantization_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("quantization_config", config.get("quantization"))
    if value is None and isinstance(config.get("text_config"), dict):
        text = config["text_config"]
        value = text.get("quantization_config", text.get("quantization"))
    from mlx_one.core.quantization import quantization_config

    result = quantization_config(value)
    if "quantization" in config and "quantization_config" in config:
        if quantization_config(config["quantization"]) != result:
            raise ValueError("conflicting quantization and quantization_config maps")
    return result


def _prepare_quantized_model(
    model: Any, tensors: dict[str, Any], quantization: dict[str, Any]
) -> None:
    if quantization.get("mode") == "turboquant":
        from mlx_one.quantization.turboquant.loading import prepare_tq3_model

        prepare_tq3_model(model, tensors, quantization)
        return

    import mlx.nn as nn

    from mlx_one.core.quantization import UnsupportedQuantization, module_quantization

    prefixes = {name.removesuffix(".scales") for name in tensors if name.endswith(".scales")}
    if not prefixes:
        raise ValueError("quantized checkpoint has no quantization scale tensors")
    modules = dict(model.named_modules())
    for path, spec in quantization.items():
        if isinstance(spec, dict) or spec is False:
            if path not in modules or not hasattr(modules[path], "to_quantized"):
                raise ValueError(f"quantization override references unknown module: {path}")
            if spec and path not in prefixes:
                raise ValueError(f"missing quantization scale tensors for override: {path}")
    formats = [module_quantization(quantization, path) for path in prefixes]
    if any(spec and spec["mode"] == "mxfp4" for spec in formats):
        _check_mxfp4_core()

    def predicate(path, module):
        if path not in prefixes:
            return False
        spec = module_quantization(quantization, path)
        if spec is False or not hasattr(module, "to_quantized"):
            raise UnsupportedQuantization(f"packed tensors on non-quantized module: {path}")
        return spec

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization["mode"],
        class_predicate=predicate,
    )


def _check_mxfp4_core():
    """Probe the actual core operation, never interpret MXFP4 payloads as affine."""
    import mlx.core as mx

    from mlx_one.core.quantization import UnsupportedQuantization

    try:
        weights, scales = mx.quantize(mx.zeros((1, 32, 32)), group_size=32, bits=4, mode="mxfp4")
        result = mx.gather_qmm(mx.zeros((1, 1, 32)), weights, scales,
                               rhs_indices=mx.array([0]), group_size=32, bits=4, mode="mxfp4")
        mx.eval(result)
    except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
        raise UnsupportedQuantization(
            "UNSUPPORTED_QUANTIZATION: INCOMPATIBLE_MLX_CORE: native MXFP4 gather_qmm "
            "is unavailable. Install a compatible Apple Silicon pair: "
            "python -m pip install --upgrade 'mlx==0.32.2' 'mlx-metal==0.32.2'. "
            "MXFP4 cannot be loaded as affine Q4."
        ) from exc


def _check_checkpoint_memory(root):
    """Reject weights larger than the detected Metal working-set headroom.

    This is a detectable lower bound; runtime admission still owns KV/transients.
    No promise about practical context size follows from the configured maximum.
    """
    import mlx.core as mx

    files, _ = _checkpoint_files(root)
    info = mx.device_info()
    recommended = info.get("max_recommended_working_set_size")
    if recommended and sum(path.stat().st_size for path in files) > (
        recommended - mx.get_active_memory()
    ):
        raise TextModelLoadError("INSUFFICIENT_MEMORY: checkpoint exceeds detected Metal "
                                 "working-set headroom; unload the active model first")


def _validate_quantized_tensors(model: Any, tensors: dict[str, Any]) -> None:
    """Validate mixed quantized parameters against the transformed model tree."""
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(tensors)
        if tuple(expected[name].shape) != tuple(tensors[name].shape)
    )
    named = getattr(model, "named_modules", lambda: ())
    for path, module in named():
        if hasattr(module, "bits"):
            _validate_packed_module(path, module, tensors)
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:5]}")
        raise ValueError("invalid mixed quantized checkpoint: " + ", ".join(details))


def _validate_packed_module(path, module, tensors):
    import mlx.core as mx

    prefix = path + "." if path else ""
    if not all(prefix + name in tensors for name in ("weight", "scales")):
        return  # Missing parameters are reported together by the caller.
    weight, scales = (tensors[prefix + name] for name in ("weight", "scales"))
    if weight.dtype != mx.uint32 or weight.ndim not in {2, 3}:
        raise ValueError(f"{path}: packed weights must be rank-two/three uint32")
    if module.mode == "mxfp4":
        if scales.dtype != mx.uint8:
            raise ValueError(f"{path}: MXFP4 scales must use uint8 E8M0 storage")
        if prefix + "biases" in tensors:
            raise ValueError(f"{path}: MXFP4 has no affine quantization biases")
    elif module.mode == "affine":
        biases = tensors.get(prefix + "biases")
        if biases is not None and (scales.dtype not in {mx.float16, mx.bfloat16, mx.float32}
                                   or biases.dtype != scales.dtype):
            raise ValueError(f"{path}: affine scale/bias tensors need matching floating dtype")


def _context_length(config: Any) -> int:
    if hasattr(config, "text_config"):
        config = config.text_config
    for name in ("max_position_embeddings", "max_context_length", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _parameter_count(config: dict[str, Any]) -> int | None:
    for name in ("num_parameters", "parameter_count", "n_params"):
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _eos_token_ids(config: dict[str, Any]) -> tuple[int, ...]:
    value = config.get("eos_token_id")
    if value is None and isinstance(config.get("text_config"), dict):
        value = config["text_config"].get("eos_token_id")
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    if not values or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values
    ):
        raise ValueError("eos_token_id must be a non-negative integer or list of integers")
    return tuple(dict.fromkeys(values))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextModelLoadError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TextModelLoadError(f"{path.name} must contain a JSON object")
    return value


def _read_safetensors(root: Path) -> dict[str, Any]:
    files, expected_map = _checkpoint_files(root)

    result: dict[str, Any] = {}
    try:
        for path in files:
            tensors = _load_safetensor_file(path)
            for name, tensor in tensors.items():
                if name in result:
                    raise TextModelLoadError(f"duplicate text tensor: {name}")
                if expected_map is not None and expected_map.get(name) != path.name:
                    raise TextModelLoadError(f"safetensors index mismatch for tensor: {name}")
                result[name] = tensor
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, TextModelLoadError):
            raise
        raise TextModelLoadError(f"cannot read text safetensors: {exc}") from exc
    if expected_map is not None and set(result) != set(expected_map):
        raise TextModelLoadError("safetensors index and shard tensors do not match")
    return result


def _load_safetensor_file(path: Path) -> dict[str, Any]:
    """Load safetensors directly with MLX so BF16 never passes through NumPy."""

    from mlx_one.core.precision import validate_finite_safetensors

    validate_finite_safetensors(path)
    import mlx.core as mx

    tensors = mx.load(str(path))
    if not isinstance(tensors, dict):
        raise ValueError(f"{path.name} did not contain a tensor mapping")
    return tensors


def _checkpoint_files(root: Path) -> tuple[tuple[Path, ...], dict[str, str] | None]:
    index_path = root / "model.safetensors.index.json"
    files = tuple(sorted(root.glob("*.safetensors")))
    if not files:
        unsafe = tuple(root.glob("*.bin")) + tuple(root.glob("*.pt"))
        detail = "; pickle checkpoints are not accepted" if unsafe else ""
        raise TextModelLoadError(f"native text safetensors checkpoint is missing{detail}")
    expected_map: dict[str, str] | None = None
    if index_path.exists():
        index = _read_json(index_path)
        value = index.get("weight_map")
        if not isinstance(value, dict) or not value:
            raise TextModelLoadError("safetensors index requires a non-empty weight_map")
        if any(
            not isinstance(name, str) or not isinstance(file, str) for name, file in value.items()
        ):
            raise TextModelLoadError("safetensors weight_map must map tensor names to files")
        expected_map = value
        referenced = {root / file for file in value.values()}
        if any(path.parent != root or not path.is_file() for path in referenced):
            raise TextModelLoadError("safetensors index references a missing or unsafe shard")
        files = tuple(sorted(referenced))
    elif len(files) != 1:
        raise TextModelLoadError("multiple safetensors files require an index")
    resolved_root = root.resolve()
    # Hub snapshots intentionally link into their own repository blob store.
    # Arbitrary local symlinks and non-safetensors index targets are not weights.
    cache_blobs = (root.parent.parent / "blobs").resolve()
    hub_snapshot = (root.parent.name == "snapshots"
                    and root.parent.parent.name.startswith("models--")
                    and re.fullmatch(r"[0-9a-f]{40,64}", root.name))
    if any(path.suffix != ".safetensors" or (
        path.resolve().parent != resolved_root
        and not (hub_snapshot and path.resolve().parent == cache_blobs)
    ) for path in files):
        raise TextModelLoadError("safetensors checkpoint references an unsafe shard")
    return files, expected_map
