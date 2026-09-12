"""Native batch embedding and reranking services."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from mlx_one.retrieval.loading import LoadedRetrievalModel, load_retrieval_model
from mlx_one.retrieval.schemas import EmbeddingResult, RerankItem, RerankResult

DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def embed(
    model: str | Path | LoadedRetrievalModel,
    texts: str | Sequence[str],
    *,
    input_type: Literal["query", "document"] = "document",
    instruction: str | None = None,
    dimensions: int | None = None,
    max_length: int = 8192,
    batch_size: int = 8,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> EmbeddingResult:
    bundle = _bundle(
        model,
        "embedding",
        revision=revision,
        offline=offline,
        cache_dir=cache_dir,
    )
    values = (texts,) if isinstance(texts, str) else tuple(texts)
    _validate_texts(values, "embedding inputs")
    if input_type not in {"query", "document"}:
        raise ValueError("input_type must be query or document")
    if input_type == "document" and instruction is not None:
        raise ValueError("instruction is only valid for query embeddings")
    _validate_batch_options(max_length, batch_size, bundle.model.config.max_seq_length)
    size = bundle.model.config.validate_dimensions(dimensions)
    task = DEFAULT_RETRIEVAL_INSTRUCTION if instruction is None else instruction
    if input_type == "query" and (not isinstance(task, str) or not task):
        raise ValueError("query instruction must be non-empty text")
    formatted = (
        tuple(f"Instruct: {task}\nQuery:{value}" for value in values)
        if input_type == "query"
        else values
    )
    vectors: list[tuple[float, ...]] = []
    import mlx.core as mx

    for start in range(0, len(formatted), batch_size):
        sequences = [
            bundle.tokenizer.encode(value, add_special_tokens=True, max_length=max_length)
            for value in formatted[start : start + batch_size]
        ]
        ids, masks = bundle.tokenizer.pad(sequences, padding_side="left")
        output = bundle.model(
            mx.array(ids), attention_mask=mx.array(masks), dimensions=size
        )
        mx.eval(output.embeddings)
        vectors.extend(tuple(float(item) for item in row) for row in output.embeddings.tolist())
    return EmbeddingResult(
        values,
        tuple(vectors),
        size,
        input_type,
        bundle.model_id,
        bundle.revision,
    )


def rerank(
    model: str | Path | LoadedRetrievalModel,
    query: str,
    documents: Sequence[str],
    *,
    instruction: str | None = None,
    top_k: int | None = None,
    max_length: int = 8192,
    batch_size: int = 8,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> RerankResult:
    bundle = _bundle(
        model,
        "reranking",
        revision=revision,
        offline=offline,
        cache_dir=cache_dir,
    )
    if not isinstance(query, str) or not query:
        raise ValueError("query must be non-empty text")
    if isinstance(documents, str):
        raise ValueError("documents must be a sequence of strings, not one string")
    values = tuple(documents)
    _validate_texts(values, "documents")
    _validate_batch_options(max_length, batch_size, bundle.model.config.max_seq_length)
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k must be null or a positive integer")
    task = DEFAULT_RETRIEVAL_INSTRUCTION if instruction is None else instruction
    if not isinstance(task, str) or not task:
        raise ValueError("reranking instruction must be non-empty text")
    prefix = bundle.tokenizer.encode(RERANK_PREFIX)
    suffix = bundle.tokenizer.encode(RERANK_SUFFIX)
    capacity = max_length - len(prefix) - len(suffix)
    if capacity < 1:
        raise ValueError("max_length leaves no room for the reranking pair")
    true_id = _single_token(bundle, "yes")
    false_id = _single_token(bundle, "no")
    results: list[RerankItem] = []
    import mlx.core as mx

    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        sequences = []
        for document in batch:
            core = f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {document}"
            sequences.append(prefix + bundle.tokenizer.encode(core)[:capacity] + suffix)
        ids, masks = bundle.tokenizer.pad(sequences, padding_side="left")
        output = bundle.model(
            mx.array(ids),
            attention_mask=mx.array(masks),
            false_token_id=false_id,
            true_token_id=true_id,
        )
        mx.eval(output.scores, output.probabilities)
        for offset, (score, probability) in enumerate(
            zip(output.scores.tolist(), output.probabilities.tolist(), strict=True)
        ):
            index = start + offset
            results.append(
                RerankItem(index, values[index], float(score), float(probability))
            )
    results.sort(key=lambda item: (-item.score, item.index))
    if top_k is not None:
        results = results[:top_k]
    return RerankResult(query, tuple(results), bundle.model_id, bundle.revision)


def _bundle(
    model: str | Path | LoadedRetrievalModel,
    task: Literal["embedding", "reranking"],
    *,
    revision: str | None,
    offline: bool,
    cache_dir: str | Path | None,
) -> LoadedRetrievalModel:
    bundle = (
        model
        if isinstance(model, LoadedRetrievalModel)
        else load_retrieval_model(
            model,
            task=task,
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
        )
    )
    if bundle.task != task:
        raise ValueError(f"loaded retrieval bundle is for {bundle.task}, not {task}")
    return bundle


def _single_token(bundle: LoadedRetrievalModel, text: str) -> int:
    encoded = bundle.tokenizer.encode(text)
    if len(encoded) != 1:
        raise ValueError(f"reranker answer {text!r} must encode to one token")
    return encoded[0]


def _validate_texts(values: tuple[str, ...], label: str) -> None:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain non-empty strings")


def _validate_batch_options(max_length: int, batch_size: int, capacity: int) -> None:
    for name, value in (("max_length", max_length), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_length > capacity:
        raise ValueError("max_length exceeds the model configuration")
