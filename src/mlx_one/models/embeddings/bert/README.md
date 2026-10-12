# BERT / all-MiniLM-L6-v2

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `bert`. The native encoder implements bidirectional attention,
absolute positions, token-type embeddings, mean pooling, normalization, and
single-vector retrieval. The primary checkpoint candidate is
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Native loading and synthetic validation exist; real-checkpoint parity, retrieval
quality, and hardware throughput still need qualification.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Token IDs + token-type IDs + position IDs
                    ↓
       Summed embeddings → LayerNorm
                    ↓
┌──────────────── Bidirectional encoder × L ────────────────┐
│ Multi-head self-attention → residual + LayerNorm          │
│ Feed-forward + activation → residual + LayerNorm          │
└───────────────────────────────────────────────────────────┘
                    ↓
          Padding-aware masked mean pooling
                    ↓
     Optional L2 normalization → one vector per text
                    └→ Cosine similarity for sentence pairs
```

## Papers and upstream references

Papers: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805),
[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957),
and [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).

BERT supplies the encoder topology; MiniLM describes the compact distilled
backbone used by the MiniLM candidate. Sentence-BERT is the sentence-embedding
reference, not proof that every BERT checkpoint is a trained sentence encoder.

## Installation and assets

Use Python 3.10+ and `python -m pip install -e .` from the repository root.
Execution requires Apple Silicon/Metal. Preserve `config.json`, complete
safetensors weights/index, `tokenizer.json`, tokenizer metadata,
`sentence_bert_config.json`, and `1_Pooling/config.json`. A plain BERT language-model
checkpoint is not interchangeable with the Sentence Transformer bundle.

## CLI and Python examples

Use plain sentences without chat markers. The candidate examples may download
weights; add an immutable `revision` for reproducible runs.

```bash
mlx-one inspect sentence-transformers/all-MiniLM-L6-v2
mlx-one embed sentence-transformers/all-MiniLM-L6-v2 "MLX uses unified memory." \
  --max-length 256
```

```python
from mlx_one import embed

result = embed(
    "sentence-transformers/all-MiniLM-L6-v2",
    ["MLX uses unified memory.", "A recipe for sourdough."],
    max_length=256,
    batch_size=2,
)
first, second = result.embeddings
similarity = sum(left * right for left, right in zip(first, second))
print(result.dimensions, similarity)
```

With normalized outputs, the dot product equals cosine similarity. Each input
produces one vector; this is distinct from ColBERT's token-vector representation.

## Settings and troubleshooting

- Set `max_length` explicitly: the generic service defaults to 8192, above this
  checkpoint's typical 256-token sentence limit. The loaded config is authoritative.
- The native profile is 384 dimensions; arbitrary Matryoshka truncation is not
  supported here. `dimensions` must be absent or equal the native width.
- Query instruction formatting is implemented for Qwen3 embeddings, not this
  encoder. Supply ordinary query/document text.
- Pooling and sentence metadata must match the model. Missing files or non-absolute
  position settings require a compatible bundle.
- This is an encoder workflow. Autoregressive KV, chat, APC, and block decoding
  are not part of its embedding task.

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
