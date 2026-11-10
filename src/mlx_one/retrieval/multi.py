"""Native BGE-M3 dense/sparse/token-vector output and explicit hybrid scoring."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_one.retrieval.loading import LoadedRetrievalModel, load_retrieval_model
from mlx_one.retrieval.service import _validate_batch_options, _validate_texts


@dataclass(frozen=True)
class MultiRepresentationResult:
    texts: tuple[str, ...]
    dense_embeddings: tuple
    sparse_weights: tuple[dict[str, float], ...]
    token_embeddings: tuple
    token_ids: tuple
    model: str
    revision: str | None
    execution_identity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


def encode_multi(
    model: str | Path | LoadedRetrievalModel,
    texts: str | Sequence[str],
    *,
    max_length: int = 512,
    batch_size: int = 1,
    revision: str | None = None,
    offline: bool = False,
) -> MultiRepresentationResult:
    """Return all three trained BGE-M3 modes; source profiles without heads fail."""
    bundle = (
        model
        if isinstance(model, LoadedRetrievalModel)
        else load_retrieval_model(model, task="embedding", revision=revision, offline=offline)
    )
    if bundle.architecture != "xlm-roberta" or not bundle.model.config.multi_representation:
        raise ValueError("multi-representation requires the complete converted BGE-M3 checkpoint")
    values = (texts,) if isinstance(texts, str) else tuple(texts)
    _validate_texts(values, "multi-representation inputs")
    _validate_batch_options(max_length, batch_size, bundle.model.config.sentence_max_length)
    skip = {
        bundle.tokenizer._tokenizer.token_to_id(token)
        for token in ("<s>", "</s>", "<pad>", "<unk>")
    }
    dense, sparse, vectors, token_ids = [], [], [], []
    import mlx.core as mx

    for start in range(0, len(values), batch_size):
        sequences = [
            bundle.tokenizer.encode(t, add_special_tokens=True, max_length=max_length)
            for t in values[start : start + batch_size]
        ]
        if len(sequences) * max(map(len, sequences)) > 8192:
            raise ValueError("BGE-M3 eager attention is bounded to 8192 actual batch tokens")
        ids, masks = bundle.tokenizer.pad(sequences)
        output = bundle.model(mx.array(ids), attention_mask=mx.array(masks), representations=True)
        mx.eval(output.embeddings, output.sparse_weights, output.colbert_embeddings)
        dense.extend(tuple(row) for row in output.embeddings.tolist())
        for tokens, weights, features in zip(
            sequences,
            output.sparse_weights.tolist(),
            output.colbert_embeddings.tolist(),
            strict=True,
        ):
            lexical = {}
            for token, weight in zip(tokens, weights[: len(tokens)], strict=True):
                if token not in skip and weight > 0:
                    lexical[str(token)] = max(lexical.get(str(token), 0.0), weight)
            sparse.append(lexical)
            vectors.append(tuple(tuple(row) for row in features[: len(tokens) - 1]))
            token_ids.append(tuple(tokens[1:]))
    identity = hashlib.sha256(
        repr((id(bundle.model), bundle.model_id, bundle.revision)).encode()
    ).hexdigest()
    return MultiRepresentationResult(
        values,
        tuple(dense),
        tuple(sparse),
        tuple(vectors),
        tuple(token_ids),
        bundle.model_id,
        bundle.revision,
        identity,
    )


def hybrid_scores(
    query: MultiRepresentationResult,
    documents: MultiRepresentationResult,
    *,
    dense_weight: float = 1.0,
    sparse_weight: float = 0.3,
    colbert_weight: float = 1.0,
) -> tuple[dict[str, float], ...]:
    """Score one query against documents; return independent and combined scores."""
    if (
        len(query.texts) != 1
        or not documents.texts
        or query.execution_identity != documents.execution_identity
    ):
        raise ValueError("hybrid scoring needs one query and features from the same loaded model")
    weights = np.asarray([dense_weight, sparse_weight, colbert_weight], float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("hybrid weights must be finite")
    qdense = np.asarray(query.dense_embeddings[0])
    qtokens = np.asarray(query.token_embeddings[0])
    if not len(qtokens):
        raise ValueError("query token representations must be nonempty")
    scores = []
    for i, (dense, sparse, tokens) in enumerate(
        zip(
            documents.dense_embeddings,
            documents.sparse_weights,
            documents.token_embeddings,
            strict=True,
        )
    ):
        lexical = sum(
            weight * sparse.get(token, 0.0) for token, weight in query.sparse_weights[0].items()
        )
        token_score = float((qtokens @ np.asarray(tokens).T).max(axis=-1).mean())
        dense_score = float(qdense @ np.asarray(dense))
        scores.append(
            dict(
                index=i,
                dense=dense_score,
                sparse=lexical,
                colbert=token_score,
                score=dense_weight * dense_score
                + sparse_weight * lexical
                + colbert_weight * token_score,
            )
        )
    return tuple(sorted(scores, key=lambda row: (-row["score"], row["index"])))
