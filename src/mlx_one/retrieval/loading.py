"""Safe native Qwen3 retrieval-model loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import snapshot_download
from safetensors import SafetensorError, safe_open

from mlx_one.models.embeddings.qwen3_embedding.config import Qwen3EmbeddingConfig
from mlx_one.models.embeddings.qwen3_embedding.weights import (
    sanitize_weights as sanitize_embedding_weights,
)
from mlx_one.models.embeddings.qwen3_embedding.weights import (
    weight_contract as embedding_weight_contract,
)
from mlx_one.models.embeddings.qwen3_reranker.config import Qwen3RerankerConfig
from mlx_one.models.embeddings.qwen3_reranker.weights import (
    sanitize_weights as sanitize_reranker_weights,
)
from mlx_one.models.embeddings.qwen3_reranker.weights import (
    weight_contract as reranker_weight_contract,
)
from mlx_one.retrieval.tokenizer import Qwen3Tokenizer

RetrievalTask = Literal["embedding", "reranking"]


class RetrievalModelLoadError(RuntimeError):
    """Raised when a native retrieval bundle is unsafe or incomplete."""


@dataclass(frozen=True)
class LoadedRetrievalModel:
    model: Any
    tokenizer: Qwen3Tokenizer
    task: RetrievalTask
    path: Path
    model_id: str
    revision: str | None


def load_retrieval_model(
    model: str | Path,
    *,
    task: RetrievalTask | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedRetrievalModel:
    if task not in {None, "embedding", "reranking"}:
        raise RetrievalModelLoadError(f"unsupported retrieval task: {task!r}")
    root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
    config_data = _read_json(root / "config.json")
    if config_data.get("model_type") != "qwen3":
        raise RetrievalModelLoadError(
            f"unsupported native retrieval model type: {config_data.get('model_type')!r}"
        )
    try:
        detected = detect_retrieval_task(root)
    except RetrievalModelLoadError:
        if task is None:
            raise
        detected = task
    if task is not None and task != detected:
        raise RetrievalModelLoadError(
            f"requested retrieval task {task!r} conflicts with bundle task {detected!r}"
        )
    task = detected
    try:
        tokenizer = Qwen3Tokenizer.from_directory(root)
    except ValueError as exc:
        raise RetrievalModelLoadError(f"invalid Qwen3 tokenizer assets: {exc}") from exc
    vocab_size = config_data.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise RetrievalModelLoadError("Qwen3 config requires a positive integer vocab_size")
    if max(tokenizer.encoder.values(), default=-1) >= vocab_size:
        raise RetrievalModelLoadError("Qwen3 tokenizer IDs exceed the configured vocabulary")
    tensors = _read_safetensors(root)
    if task == "embedding":
        pooling = _read_json(root / "1_Pooling" / "config.json")
        if pooling.get("pooling_mode_lasttoken") is not True:
            raise RetrievalModelLoadError("Qwen3 embedding requires last-token pooling metadata")
        pooling_dimension = pooling.get("word_embedding_dimension")
        if isinstance(pooling_dimension, bool) or not isinstance(pooling_dimension, int):
            raise RetrievalModelLoadError(
                "Qwen3 embedding pooling metadata requires word_embedding_dimension"
            )
        payload = {
            **config_data,
            "model_type": "qwen3_embedding",
            "word_embedding_dimension": pooling_dimension,
        }
        try:
            config = Qwen3EmbeddingConfig.from_dict(payload)
            tensors = sanitize_embedding_weights(tensors, config)
            embedding_weight_contract(config).validate(tensors)
        except ValueError as exc:
            raise RetrievalModelLoadError(
                f"invalid Qwen3 embedding architecture or weights: {exc}"
            ) from exc
        from mlx_one.models.embeddings.qwen3_embedding.model import Qwen3ForEmbedding

        native = Qwen3ForEmbedding(config)
    else:
        try:
            config = Qwen3RerankerConfig.from_dict(
                {**config_data, "model_type": "qwen3_reranker"}
            )
            tensors = sanitize_reranker_weights(tensors, config)
            reranker_weight_contract(config).validate(tensors)
        except ValueError as exc:
            raise RetrievalModelLoadError(
                f"invalid Qwen3 reranker architecture or weights: {exc}"
            ) from exc
        from mlx_one.models.embeddings.qwen3_reranker.model import Qwen3ForReranking

        native = Qwen3ForReranking(config)
        for answer in ("yes", "no"):
            encoded = tokenizer.encode(answer)
            if len(encoded) != 1:
                raise RetrievalModelLoadError(
                    f"Qwen3 reranker answer token {answer!r} must encode to one token"
                )
    native.load_weights(list(tensors.items()), strict=True)
    native.eval()
    resolved_revision = revision
    if resolved_revision is None and re.fullmatch(r"[0-9a-f]{40,64}", root.name):
        resolved_revision = root.name
    return LoadedRetrievalModel(native, tokenizer, task, root, str(model), resolved_revision)


def detect_retrieval_task(root: str | Path) -> RetrievalTask:
    directory = Path(root)
    metadata_path = directory / "config_sentence_transformers.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    metadata_markers = " ".join(
        str(metadata.get(name, ""))
        for name in ("model_type", "task", "architecture", "architectures")
    ).lower()
    if any(marker in metadata_markers for marker in ("crossencoder", "rerank")):
        return "reranking"
    modules_path = directory / "modules.json"
    pooling_path = directory / "1_Pooling" / "config.json"
    if modules_path.exists():
        modules = _read_json_array(modules_path)
        module_types = " ".join(
            str(item.get("type", "")) for item in modules if isinstance(item, dict)
        ).lower()
        if any(
            marker in module_types
            for marker in ("crossencoder", "cross_encoder", "rerank", "logit_score")
        ):
            return "reranking"
        if pooling_path.exists() and any(
            isinstance(item, dict)
            and item.get("type") == "sentence_transformers.models.Pooling"
            for item in modules
        ):
            return "embedding"
    raise RetrievalModelLoadError("cannot detect Qwen3 embedding or reranking task metadata")


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
            raise RetrievalModelLoadError("local retrieval model must be a directory")
        return local.resolve()
    if isinstance(model, Path) or str(model).startswith((".", "/", "~")):
        raise RetrievalModelLoadError(f"local retrieval model directory does not exist: {local}")
    try:
        return Path(
            snapshot_download(
                str(model),
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=offline,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "modules.json",
                    "config_sentence_transformers.json",
                    "1_Pooling/config.json",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
            )
        )
    except Exception as exc:
        raise RetrievalModelLoadError(f"cannot resolve native retrieval model: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalModelLoadError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RetrievalModelLoadError(f"{path.name} must contain a JSON object")
    return value


def _read_json_array(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalModelLoadError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, list):
        raise RetrievalModelLoadError(f"{path.name} must contain a JSON array")
    return value


def _read_safetensors(root: Path) -> dict[str, Any]:
    files, expected_map = _checkpoint_files(root)
    import mlx.core as mx

    result: dict[str, Any] = {}
    try:
        for path in files:
            with safe_open(path, framework="numpy") as handle:
                for name in handle.keys():
                    if name in result:
                        raise RetrievalModelLoadError(f"duplicate retrieval tensor: {name}")
                    if expected_map is not None and expected_map.get(name) != path.name:
                        raise RetrievalModelLoadError(
                            f"safetensors index mismatch for tensor: {name}"
                        )
                    result[name] = mx.array(handle.get_tensor(name))
    except (OSError, SafetensorError) as exc:
        raise RetrievalModelLoadError(f"cannot read retrieval safetensors: {exc}") from exc
    if expected_map is not None and set(result) != set(expected_map):
        raise RetrievalModelLoadError("safetensors index and shard tensors do not match")
    return result


def _checkpoint_files(root: Path) -> tuple[tuple[Path, ...], dict[str, str] | None]:
    index_path = root / "model.safetensors.index.json"
    files = tuple(sorted(root.glob("*.safetensors")))
    if not files:
        unsafe = tuple(root.glob("*.bin")) + tuple(root.glob("*.pt"))
        detail = "; pickle checkpoints are not accepted" if unsafe else ""
        raise RetrievalModelLoadError(f"native retrieval safetensors checkpoint is missing{detail}")
    expected_map: dict[str, str] | None = None
    if index_path.exists():
        index = _read_json(index_path)
        value = index.get("weight_map")
        if not isinstance(value, dict) or not value:
            raise RetrievalModelLoadError("safetensors index requires a non-empty weight_map")
        if any(
            not isinstance(name, str) or not isinstance(file, str)
            for name, file in value.items()
        ):
            raise RetrievalModelLoadError("safetensors weight_map must map tensor names to files")
        expected_map = value
        referenced = {root / file for file in value.values()}
        resolved_root = root.resolve()
        if any(
            path.parent != root
            or path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != resolved_root
            for path in referenced
        ):
            raise RetrievalModelLoadError("safetensors index references a missing or unsafe shard")
        files = tuple(sorted(referenced))
    elif len(files) != 1:
        raise RetrievalModelLoadError("multiple safetensors files require an index")
    return files, expected_map
