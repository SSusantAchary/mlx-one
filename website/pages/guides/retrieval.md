# Embeddings and reranking

Native retrieval includes single-vector embeddings, Qwen3 pairwise reranking,
and a separate ColBERT token-level path. All **five retrieval model types** have
passing real-checkpoint primary workflows in the
[current inventory](../../../status_models_testing.md). Broader retrieval quality
and throughput retain their own evidence; see the
[family index](../../../docs/models/index.md#retrieval-guides).

## Embed a query

```python
from mlx_one import embed

query = embed(
    "Qwen/Qwen3-Embedding-0.6B",
    "How does MLX use unified memory?",
    input_type="query",
    dimensions=512,
)
print(query.embeddings[0])
```

Document inputs are unprefixed. Instructed queries use the model’s documented
query format. Consult the [Qwen3 embedding guide](../../../src/mlx_one/models/embeddings/qwen3_embedding/README.md)
for dimensions, pooling, normalization and checkpoint assets.

## Rerank candidate documents

```python
from mlx_one import rerank

ranking = rerank(
    "Qwen/Qwen3-Reranker-0.6B",
    "How does MLX use unified memory?",
    ["MLX arrays share CPU and GPU memory.", "A recipe for sourdough."],
    top_k=1,
)
print(ranking.items[0].index, ranking.items[0].score)
```

The [reranker guide](../../../src/mlx_one/models/embeddings/qwen3_reranker/README.md)
describes yes/no logit scoring, probability semantics and stable ranking.

## Use the CLI

```bash
mlx-one embed Qwen/Qwen3-Embedding-0.6B "A document to index" \
  --dimensions 512 --json-output

mlx-one rerank Qwen/Qwen3-Reranker-0.6B "Which Macs support MLX?" \
  "MLX is designed for Apple silicon." "CUDA targets NVIDIA GPUs." \
  --top-k 1 --json-output
```

LFM2 ColBERT emits token vectors for masked MaxSim. It is not interchangeable
with the generic single-vector `embed` contract. See the
[ColBERT guide](../../../src/mlx_one/models/embeddings/lfm2_colbert/README.md).
