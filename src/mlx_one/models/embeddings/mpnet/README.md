# MPNet / all-mpnet-base-v2

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `mpnet`. The encoder implements MPNet relative-position buckets,
bidirectional attention, mean pooling, normalization, and single-vector retrieval.
[sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
is the primary checkpoint candidate. Native loading and synthetic execution are
implemented; exact-checkpoint quality, parity, and hardware qualification remain pending.

## Installation and assets

Install `python -m pip install -e .` from the repository root using Python 3.10+.
Execution needs Apple Silicon/Metal. Keep `config.json`, safetensors weights/index,
`tokenizer.json`, tokenizer metadata, `sentence_bert_config.json`, and
`1_Pooling/config.json`. Relative-position weights and special-token IDs must
match the MPNet configuration; a BERT checkpoint is not a substitute.

## CLI and Python examples

Use ordinary sentences. These candidate examples may download weights; pin an
immutable `revision` for reproducibility.

```bash
mlx-one inspect sentence-transformers/all-mpnet-base-v2
mlx-one embed sentence-transformers/all-mpnet-base-v2 "MLX uses unified memory." \
  --max-length 384
```

```python
from mlx_one import embed

result = embed(
    "sentence-transformers/all-mpnet-base-v2",
    ["How does unified memory work?", "CPU and GPU share one memory pool."],
    max_length=384,
    batch_size=2,
)
query, document = result.embeddings
print(sum(left * right for left, right in zip(query, document)))
```

The dot product is cosine similarity when the outputs are normalized. The service
returns one vector per input; it does not return token embeddings for late interaction.

## Settings and troubleshooting

- Set `max_length` explicitly. The native profile's sentence limit is 384 tokens,
  whereas the generic service default is 8192; the loaded config governs validation.
- The native profile is 768 dimensions. Arbitrary reduced dimensions are unsupported;
  omit `dimensions` or use the native width.
- Query instructions/chat markers are not automatically applied to this family.
- Missing pooling metadata or relative-attention tensors requires a complete
  matching Sentence Transformer snapshot.
- MPNet is an encoder workflow: generation KV, server APC, and block decoding
  are not embedding capabilities.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[retrieval loading](../../../retrieval/loading.py),
[architecture tests](../../../../../tests/test_native_sentence_transformers_architectures.py),
and [synthetic Metal tests](../../../../../tests/test_native_sentence_transformers_mlx.py).

```bash
pytest -q tests/test_native_sentence_transformers_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_sentence_transformers_mlx.py
```

See [qualification levels](../../../../../README.md#capability-levels) and the
[family backlog](../../../../../model_list.txt).
