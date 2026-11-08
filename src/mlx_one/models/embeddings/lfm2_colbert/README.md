# LFM2 ColBERT

[Model guide index](../../../../../docs/models/index.md)

## Overview and status

Registry type: `lfm2_colbert`. The native LFM2 backbone projects contextual
token vectors and scores query/document pairs using masked MaxSim late interaction.
[LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M)
is admitted by the loader only when its module pipeline, projection and token recipe
pass validation. See the [executed campaign](../../../../../docs/qualification/model-release-20261108/README.md)
for the exact checkpoint result and remaining release gates. Architecture tests
and a small sample do not establish general retrieval quality.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Query token IDs + mask                    Document token IDs + mask
          ↓                                           ↓
      LFM2 hybrid backbone (shared weights; separate forward passes)
          ↓                                           ↓
      Linear projection                           Linear projection
          ↓                                           ↓
   Optional per-token L2 norm                  Optional per-token L2 norm
          └───────────────────┬───────────────────────┘
                              ↓
            Token similarity matrix [batch, query_tokens, doc_tokens]
                              ↓
            Mask document padding → max over document tokens
                              ↓
            Mask query padding → sum maxima → MaxSim score
```

## Papers and upstream references

Papers: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
and [LFM2 Technical Report](https://arxiv.org/abs/2511.23404).

ColBERT supplies the late-interaction scoring reference; this local architecture
substitutes an LFM2 backbone for BERT. Neither paper by itself qualifies the
LiquidAI checkpoint's tokenizer, indexing recipe, or task quality.

## Installation and assets

Use Python 3.10+, `python -m pip install -e .` from the repository root, and
Apple Silicon/Metal for execution. A compatible bundle needs `model_type=lfm2_colbert`
(or the accepted hyphen alias), nested text config, projection weights, complete
safetensors weights/index, `tokenizer.json`, and matching tokenizer metadata.
The published `model_type=lfm2` layout is also accepted when `modules.json`,
`config_sentence_transformers.json` and `1_Dense/{config.json,model.safetensors}`
define the validated bias-free projection and MaxSim recipe. Keep the exact
query/document prefix strings, including their trailing spaces. Do not rename an
incompatible publisher config to bypass contract validation.

## Inspection and input contract

```bash
mlx-one inspect LiquidAI/LFM2-ColBERT-350M
```

Each text produces `[batch, tokens, projection_dim]` embeddings and a token mask,
not one pooled vector. MaxSim takes each query token's maximum similarity over
unmasked document tokens and sums those maxima, excluding masked query tokens.
The config's default lengths are 32 query tokens and 512 document tokens; callers
of the lower-level model must apply truncation and the checkpoint's query/document
formatting themselves. The task adapter applies the validated publisher prefix,
query expansion and punctuation-mask recipe. Index construction and corpus
retrieval benchmarks retain separate requirements.

## Token embeddings and MaxSim example

Use a local pinned publisher bundle. Path-based registration loads lazy MLX
weights on the service's device thread. Configure enough sequence capacity for
the query and document requests. The `embed` facade and fixed-vector HTTP route
do not accept rank-three token features.

```python
from mlx_one import BatchingConfig
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.service import InferenceService
from mlx_one.engine.task_runners.retrieval import maxsim

with_service = InferenceService(batching_config=BatchingConfig(max_sequences=4))
try:
    with_service.register_task("encoder", "/path/to/lfm2-colbert", offline=True)
    query = with_service.run_task("encoder", TaskKind.LATE_INTERACTION, {
        "texts": ["What is unified memory?"], "input_type": "query", "max_length": 128,
    }).output
    documents = with_service.run_task("encoder", TaskKind.LATE_INTERACTION, {
        "texts": ["CPU and GPU share a memory pool.", "A recipe for sourdough."],
        "input_type": "document", "max_length": 128,
    }).output
    print(maxsim(query, documents))
finally:
    with_service.close()
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

See the [family backlog](../../../../../docs/qualification/archived-evidence.md) for candidate status.
