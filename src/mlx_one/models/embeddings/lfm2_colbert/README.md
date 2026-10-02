# LFM2 ColBERT

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `lfm2_colbert`. The native LFM2 backbone projects contextual
token vectors and scores query/document pairs using masked MaxSim late interaction.
[LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M)
is an upstream candidate reference. Architecture and synthetic execution have
tests; exact publisher checkpoint compatibility, quality, and hardware remain
unqualified. The retrieval loader accepts compatible local ColBERT bundles.

## Installation and assets

Use Python 3.10+, `python -m pip install -e .` from the repository root, and
Apple Silicon/Metal for execution. A compatible bundle needs `model_type=lfm2_colbert`
(or the accepted hyphen alias), nested text config, projection weights, complete
safetensors weights/index, `tokenizer.json`, and matching tokenizer metadata.
Do not rename an incompatible publisher config to bypass contract validation.

## Inspection and input contract

```bash
mlx-one inspect LiquidAI/LFM2-ColBERT-350M
```

Each text produces `[batch, tokens, projection_dim]` embeddings and a token mask,
not one pooled vector. MaxSim takes each query token's maximum similarity over
unmasked document tokens and sums those maxima, excluding masked query tokens.
The config's default lengths are 32 query tokens and 512 document tokens; callers
of the lower-level model must apply truncation and the checkpoint's query/document
formatting themselves. There is no complete publisher-specific indexing recipe here.

## Token embeddings and MaxSim example

Replace the path with a local bundle that passes the native weight/config contract.
This example demonstrates the lower-level API, not a qualified publisher workflow.
The generic `embed` service and `mlx-one embed` CLI assume single-vector outputs
and are not compatible with ColBERT's rank-three output.

```python
import mlx.core as mx

from mlx_one import load_retrieval_model

bundle = load_retrieval_model("/path/to/lfm2-colbert", task="embedding", offline=True)
config = bundle.model.config
query_ids = bundle.tokenizer.encode("What is unified memory?", add_special_tokens=True)
document_ids = bundle.tokenizer.encode(
    "CPU and GPU share a memory pool.", add_special_tokens=True
)
query_ids = query_ids[: config.query_max_length]
document_ids = document_ids[: config.document_max_length]
query_mask = mx.ones((1, len(query_ids)), dtype=mx.int32)
document_mask = mx.ones((1, len(document_ids)), dtype=mx.int32)
query = bundle.model(mx.array([query_ids]), attention_mask=query_mask)
document = bundle.model(mx.array([document_ids]), attention_mask=document_mask)
score = bundle.model.maxsim(
    query.embeddings,
    document.embeddings,
    query_mask=query.attention_mask,
    document_mask=document.attention_mask,
)
mx.eval(score)
print(query.embeddings.shape, document.embeddings.shape, score.tolist())
```

## Limits and troubleshooting

The projection dimension defaults to 128 and is checkpoint-defined. Keep masks
when padding a batch; storing only an averaged vector loses late-interaction
semantics. Empty/all-masked documents are unsuitable for MaxSim. Index/export
metadata must preserve per-token vectors, masks, tokenizer, and model identity.

The backbone is hybrid, but this retrieval example uses full forward passes;
autoregressive block KV or APC is not a retrieval feature. A rank/float-conversion
failure in the generic embedding service indicates the wrong API, not evidence
that token embeddings can safely be pooled there.

## Implementation and validation

See [configuration](config.py), [model and MaxSim](model.py), [weights](weights.py),
[retrieval loader](../../../retrieval/loading.py),
[architecture tests](../../../../../tests/test_native_lfm_architectures.py), and
[synthetic Metal tests](../../../../../tests/test_native_lfm_mlx.py).

```bash
pytest -q tests/test_native_lfm_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_lfm_mlx.py
```

See the [family backlog](../../../../../model_list.txt) for candidate status.
