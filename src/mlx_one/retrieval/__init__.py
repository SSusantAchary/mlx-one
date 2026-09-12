"""Public native embedding and reranking APIs."""

from mlx_one.retrieval.loading import (
    LoadedRetrievalModel,
    RetrievalModelLoadError,
    load_retrieval_model,
)
from mlx_one.retrieval.schemas import EmbeddingResult, RerankItem, RerankResult
from mlx_one.retrieval.service import embed, rerank

__all__ = [
    "EmbeddingResult",
    "LoadedRetrievalModel",
    "RerankItem",
    "RerankResult",
    "RetrievalModelLoadError",
    "embed",
    "load_retrieval_model",
    "rerank",
]
