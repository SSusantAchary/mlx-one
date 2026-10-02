# Qwen3 Embedding

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen3_embedding`. The retrieval loader maps a publisher
`model_type=qwen3` checkpoint into this task-specific native implementation.
It uses final-token pooling, instruction-formatted queries, normalized vectors,
and configurable Matryoshka dimensions.
[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
is the implemented profile/candidate. Synthetic validation is available;
real-checkpoint retrieval quality, parity, and hardware results remain unqualified.

## Installation and assets

Use Python 3.10+, install `python -m pip install -e .` from the repository root,
and use Apple Silicon/Metal for execution. Keep `config.json`, complete
safetensors weights/index, `tokenizer.json`, tokenizer metadata, and
`1_Pooling/config.json` declaring last-token pooling and embedding width.
Task metadata helps distinguish the embedding bundle from a Qwen3 language model.
The current Qwen3 retrieval loader validates ordinary weights; it does not apply
the generic text loader's MLX quantization preparation to this profile.

## Query/document formatting and examples

`input_type="query"` formats text as `Instruct: {instruction}\nQuery:{text}`.
Documents remain unformatted. Do not pre-add this prefix when using the query API.
These candidate examples may download weights; pin an immutable revision for
reproducible retrieval experiments.

```bash
mlx-one inspect Qwen/Qwen3-Embedding-0.6B
mlx-one embed Qwen/Qwen3-Embedding-0.6B "What is unified memory?" \
  --input-type query --dimensions 512 --max-length 512
```

```python
from mlx_one import embed, load_retrieval_model

bundle = load_retrieval_model("Qwen/Qwen3-Embedding-0.6B", task="embedding")
query = embed(
    bundle,
    "What is unified memory?",
    input_type="query",
    instruction="Retrieve passages that answer the question",
    dimensions=512,
    max_length=512,
)
documents = embed(
    bundle,
    ["CPU and GPU share memory.", "A recipe for sourdough."],
    dimensions=512,
    max_length=512,
)
scores = [
    sum(left * right for left, right in zip(query.embeddings[0], vector))
    for vector in documents.embeddings
]
print(scores)
```

Reusing the loaded bundle avoids reloading weights between query and document calls.
Use the same dimensions and model revision throughout an index.

## Settings and troubleshooting

The implemented 0.6B profile accepts dimensions from 32 through 1024; the default
is its full width. Its configured sequence maximum is 32768, while the service
defaults to 8192. Neither value is a measured safe workload on every host.
Batching uses padding-aware last-token pooling; start with short inputs/small batches.

Missing last-token pooling metadata, inconsistent vocabulary IDs, or a conflicting
task indicates the wrong/incomplete bundle. A generic Qwen3 chat model is not this
embedding checkpoint. This task produces one vector per text, not generated text;
APC and block decoding are not exposed by `embed`.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[retrieval service](../../../retrieval/service.py),
[backend-free tests](../../../../../tests/test_native_qwen3_retrieval.py), and
[synthetic Metal tests](../../../../../tests/test_native_qwen3_retrieval_mlx.py).

```bash
pytest -q tests/test_native_qwen3_retrieval.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen3_retrieval_mlx.py
```

See [qualification levels](../../../../../README.md#capability-levels) before
claiming measured retrieval performance or quality.
