"""Safe native loading for registered embedding and reranking models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import snapshot_download
from safetensors import SafetensorError

from mlx_one.core.registry import get_registration
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
from mlx_one.text.loading import (
    _load_safetensor_file,
    _prepare_quantized_model,
    _quantization_config,
    _validate_quantized_tensors,
)
from mlx_one.text.tokenizers import HFTokenizerAdapter

RetrievalTask = Literal["embedding", "reranking"]


class RetrievalModelLoadError(RuntimeError):
    """Raised when a native retrieval bundle is unsafe or incomplete."""


@dataclass(frozen=True)
class LoadedRetrievalModel:
    model: Any
    tokenizer: Any
    task: RetrievalTask
    path: Path
    model_id: str
    revision: str | None
    architecture: str = "qwen3"
    query_prefix: str = ""
    document_prefix: str = ""


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
    model_type = str(config_data.get("model_type", ""))
    if model_type == "lfm2" and (root / "modules.json").is_file():
        config_data = _published_colbert_config(root, config_data)
        model_type = "lfm2_colbert"
    if model_type in {"lfm2_colbert", "lfm2-colbert"}:
        if task not in {None, "embedding"}:
            raise RetrievalModelLoadError("LFM2 ColBERT supports embedding only")
        return _load_colbert(root, model, revision, config_data)
    if model_type in {"bert", "mpnet", "xlm-roberta", "nomic_bert"}:
        detected = detect_retrieval_task(root)
        if task is not None and task != detected:
            raise RetrievalModelLoadError("requested retrieval task conflicts with checkpoint head")
        if detected == "reranking" and model_type not in {"bert", "xlm-roberta"}:
            raise RetrievalModelLoadError(f"{model_type} has no native sequence-pair scorer")
        return _load_sentence_encoder(root, model, revision, model_type, config_data, detected)
    if model_type != "qwen3":
        raise RetrievalModelLoadError(f"unsupported native retrieval model type: {model_type!r}")
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
            config = Qwen3RerankerConfig.from_dict({**config_data, "model_type": "qwen3_reranker"})
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
    return LoadedRetrievalModel(
        native,
        tokenizer,
        task,
        root,
        str(model),
        resolved_revision,
        "qwen3_embedding" if task == "embedding" else "qwen3_reranker",
    )


def _load_sentence_encoder(
    root: Path,
    model: str | Path,
    revision: str | None,
    model_type: str,
    config_data: dict[str, Any],
    task: RetrievalTask = "embedding",
) -> LoadedRetrievalModel:
    try:
        quantization = _quantization_config(config_data)
        payload = (
            {
                **config_data,
                "task": "reranking",
                "num_labels": len(config_data.get("id2label", {})),
                "sentence_max_length": config_data.get("max_position_embeddings", 512)
                - (config_data.get("pad_token_id", 1) + 1 if model_type == "xlm-roberta" else 0),
            }
            if task == "reranking"
            else _sentence_encoder_config(root, config_data)
        )
        registration = get_registration(model_type)
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
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RetrievalModelLoadError(f"cannot load native {model_type} embedding: {exc}") from exc
    resolved_revision = revision
    if resolved_revision is None and re.fullmatch(r"[0-9a-f]{40,64}", root.name):
        resolved_revision = root.name
    prefixes = _checkpoint_prefixes(root, str(model))
    return LoadedRetrievalModel(
        native, tokenizer, task, root, str(model), resolved_revision, model_type, *prefixes
    )


def _checkpoint_prefixes(root: Path, model_id: str) -> tuple[str, str]:
    """Read explicit local formatting metadata, or the selected Hub profile."""
    metadata = root / "mlx_one_retrieval.json"
    if metadata.is_file():
        value = _read_json(metadata)
        pair = value.get("query_prefix", ""), value.get("document_prefix", "")
        if not all(isinstance(prefix, str) for prefix in pair):
            raise RetrievalModelLoadError("retrieval prefixes must be strings")
        return pair
    # Formatting is checkpoint policy, not architecture selection. A local copy
    # outside the Hub cache carries its explicit mlx_one_retrieval.json sidecar.
    hub_name = "models--intfloat--multilingual-e5-small"
    if model_id == "intfloat/multilingual-e5-small" or hub_name in root.parts:
        return "query: ", "passage: "
    if (
        model_id == "nomic-ai/nomic-embed-text-v1.5"
        or "models--nomic-ai--nomic-embed-text-v1.5" in root.parts
    ):
        return "search_query: ", "search_document: "
    return "", ""


def _sentence_encoder_config(root: Path, config_data: dict[str, Any]) -> dict[str, Any]:
    """Honor supported Sentence Transformer pooling and normalization metadata."""
    pooling = _read_json(root / "1_Pooling" / "config.json")
    modes = [key for key, value in pooling.items() if key.startswith("pooling_mode_") and value]
    if (
        len(modes) != 1
        or modes[0] not in {"pooling_mode_mean_tokens", "pooling_mode_cls_token"}
        or any(
            type(value) is not bool
            for key, value in pooling.items()
            if key.startswith("pooling_mode_")
        )
    ):
        raise ValueError("native sentence encoders require exactly one mean or CLS pooling mode")
    if config_data.get("model_type") not in {"bert", "xlm-roberta"} and modes != [
        "pooling_mode_mean_tokens"
    ]:
        raise ValueError("CLS pooling is not implemented for this sentence encoder")
    dimension = pooling.get("word_embedding_dimension")
    if type(dimension) is not int or dimension != config_data.get("hidden_size"):
        raise ValueError("pooling dimension differs from the native encoder width")
    normalize = config_data.get("normalize_embeddings", True)
    if (root / "modules.json").is_file():
        modules = _read_json_array(root / "modules.json")
        types = [module.get("type") if isinstance(module, dict) else None for module in modules]
        prefix = [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
        ]
        if types not in (prefix, [*prefix, "sentence_transformers.models.Normalize"]):
            raise ValueError("unsupported Sentence Transformer module pipeline")
        if modules[1].get("path") != "1_Pooling":
            raise ValueError("pooling module path must match the loaded pooling metadata")
        normalize = len(types) == 3
    sentence = _read_json(root / "sentence_bert_config.json")
    return {
        **config_data,
        "sentence_max_length": sentence.get(
            "max_seq_length", config_data.get("max_position_embeddings", 512)
        ),
        "add_pooling_layer": False,
        "normalize_embeddings": normalize,
        **(
            {"pooling": "cls" if modes == ["pooling_mode_cls_token"] else "mean"}
            if config_data.get("model_type") == "bert"
            else {}
        ),
    }


def _published_colbert_config(root: Path, backbone: dict[str, Any]) -> dict[str, Any]:
    modules = _read_json_array(root / "modules.json")
    if (
        len(modules) != 2
        or any(not isinstance(m, dict) for m in modules)
        or [m.get("type") for m in modules]
        != ["sentence_transformers.models.Transformer", "pylate.models.Dense.Dense"]
        or [m.get("path") for m in modules] != ["", "1_Dense"]
    ):
        raise RetrievalModelLoadError("unsupported publisher ColBERT module pipeline")
    dense = _read_json(root / "1_Dense/config.json")
    if (
        dense.get("in_features") != backbone.get("hidden_size")
        or dense.get("bias") is not False
        or dense.get("activation_function") != "torch.nn.modules.linear.Identity"
    ):
        raise RetrievalModelLoadError("unsupported publisher ColBERT projection")
    recipe = _read_json(root / "config_sentence_transformers.json")
    if (
        recipe.get("similarity_fn_name") != "MaxSim"
        or recipe.get("prompts")
        or recipe.get("default_prompt_name") is not None
    ):
        raise RetrievalModelLoadError("unsupported publisher ColBERT scoring/prompt recipe")
    required = {
        "query_prefix",
        "document_prefix",
        "query_length",
        "document_length",
        "do_query_expansion",
        "attend_to_expansion_tokens",
        "skiplist_words",
    }
    if not required <= recipe.keys():
        raise RetrievalModelLoadError("incomplete publisher ColBERT token recipe")
    return dict(
        model_type="lfm2_colbert",
        text_config=backbone,
        projection_dim=dense.get("out_features"),
        normalize=True,
        query_max_length=recipe["query_length"],
        document_max_length=recipe["document_length"],
        query_prefix=recipe["query_prefix"],
        document_prefix=recipe["document_prefix"],
        do_query_expansion=recipe["do_query_expansion"],
        attend_to_expansion_tokens=recipe["attend_to_expansion_tokens"],
        skiplist_words=recipe["skiplist_words"],
        publisher_colbert=True,
    )


def _load_colbert(
    root: Path,
    model: str | Path,
    revision: str | None,
    config_data: dict[str, Any],
) -> LoadedRetrievalModel:
    try:
        quantization = _quantization_config(config_data)
        registration = get_registration("lfm2_colbert")
        config = registration.config_class().from_dict(config_data)
        tokenizer = HFTokenizerAdapter.from_directory(root)
        tensors = _read_safetensors(root)
        if config_data.get("publisher_colbert"):
            tensors = {
                (
                    "model." + name
                    if name.startswith(("embed_tokens.", "layers.", "embedding_norm."))
                    else name
                ): value
                for name, value in tensors.items()
            }
            tensors = {
                (
                    name.removesuffix(".weight") if name.endswith(".conv.conv.weight") else name
                ): value
                for name, value in tensors.items()
            }
            projection = _read_safetensors(root / "1_Dense")
            if set(projection) != {"linear.weight"}:
                raise ValueError("publisher ColBERT requires exactly one bias-free projection")
            if "projection.weight" in tensors:
                raise ValueError("duplicate publisher ColBERT projection tensors")
            tensors["projection.weight"] = projection["linear.weight"]
        tensors = registration.sanitizer()(tensors, config)
        native = registration.model_class()(config)
        if quantization:
            _prepare_quantized_model(native, tensors, quantization)
            _validate_quantized_tensors(native, tensors)
        else:
            registration.weight_contract()(config).validate(tensors)
        native.load_weights(list(tensors.items()), strict=True)
        native.eval()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RetrievalModelLoadError(f"cannot load native LFM2 ColBERT: {exc}") from exc
    resolved_revision = revision
    if resolved_revision is None and re.fullmatch(r"[0-9a-f]{40,64}", root.name):
        resolved_revision = root.name
    return LoadedRetrievalModel(
        native,
        tokenizer,
        "embedding",
        root,
        str(model),
        resolved_revision,
        "lfm2_colbert",
    )


def detect_retrieval_task(root: str | Path) -> RetrievalTask:
    directory = Path(root)
    config_path = directory / "config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    if config.get("model_type") in {"bert", "xlm-roberta"} and any(
        str(name).endswith("ForSequenceClassification") for name in config.get("architectures", [])
    ):
        labels = config.get("num_labels", len(config.get("id2label", {})))
        if labels != 1:
            raise RetrievalModelLoadError(
                "cross-encoder reranking requires a scalar classification head"
            )
        return "reranking"
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
            isinstance(item, dict) and item.get("type") == "sentence_transformers.models.Pooling"
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
                    "sentence_bert_config.json",
                    "preprocessor_config.json",
                    "mlx_one_retrieval.json",
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
    result: dict[str, Any] = {}
    try:
        for path in files:
            for name, tensor in _load_safetensor_file(path).items():
                if name in result:
                    raise RetrievalModelLoadError(f"duplicate retrieval tensor: {name}")
                if expected_map is not None and expected_map.get(name) != path.name:
                    raise RetrievalModelLoadError(f"safetensors index mismatch for tensor: {name}")
                result[name] = tensor
    except (OSError, SafetensorError, ValueError) as exc:
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
            not isinstance(name, str) or not isinstance(file, str) for name, file in value.items()
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
