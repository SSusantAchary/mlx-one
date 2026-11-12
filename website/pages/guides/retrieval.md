# Embeddings and reranking

Native retrieval covers **seven architecture types**: BERT, MPNet, Qwen3
Embedding, Qwen3 Reranker, LFM2 ColBERT, XLM-R and Nomic BERT. Checkpoint profiles
include BGE-small, multilingual MiniLM, multilingual E5, BGE-M3 and trained paired
cross-encoders. The [model workflows](../../../docs/models/batch-1-sub-6b/README.md)
document pooling, prefixes, Matryoshka dimensions and dense/sparse/token outputs.
Broader retrieval quality
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

That format applies to Qwen3 Embedding. Multilingual E5 uses `query: ` / `passage: `
prefixes, and Nomic uses `search_query: ` / `search_document: `. BGE-small and
multilingual MiniLM have their own pooling and normalization profiles. Use the
[checkpoint-specific retrieval guide](../../../docs/models/batch-1-sub-6b/README.md#pooling-and-prefixes)
when changing model families.

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

BGE-M3 also supplies trained sparse weights and token embeddings, plus hybrid
scoring through `encode_multi()` and `hybrid_scores()`. Those extra representations
use Python/CLI; dense embeddings have the tested Engine/HTTP path. See
[BGE-M3's three representations](../../../docs/models/batch-1-sub-6b/README.md#bge-m3s-three-representations).
