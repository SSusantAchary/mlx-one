# Qwen3 Reranker

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen3_reranker`. The task-specific native Qwen3 decoder scores
query/document pairs using yes/no logits rather than producing embedding vectors.
The implemented profile/candidate is
[Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
Prompt construction, scoring, probabilities, and stable ranking have synthetic
tests; real-checkpoint quality, parity, and hardware performance remain unqualified.

## Installation and assets

Install `python -m pip install -e .` from the repository root using Python 3.10+.
Execution requires Apple Silicon/Metal. Keep publisher `model_type=qwen3` config,
complete safetensors weights/index, `tokenizer.json`, tokenizer metadata, and
task-identifying assets. The loader requires `yes` and `no` to encode as single
tokens. The Qwen3 retrieval path validates ordinary weights rather than preparing
MLX quantized modules like the text loader does.

## Pair prompts and Python example

The service builds the official pair structure with an instruction, `Query`, and
`Document`, including the assistant suffix. Pass plain strings rather than a
preformatted chat prompt. It computes pair scores without free-form answer generation.

This candidate example may download weights; use an immutable `revision` for
reproducible evaluation.

```python
from mlx_one import rerank

result = rerank(
    "Qwen/Qwen3-Reranker-0.6B",
    "What is unified memory?",
    ["CPU and GPU share one memory pool.", "A recipe for sourdough."],
    instruction="Rank documents by whether they answer the question",
    top_k=2,
    max_length=512,
    batch_size=2,
)
for item in result.items:
    print(item.index, item.score, item.probability, item.document)
```

## CLI examples

Supply each candidate document as a separate positional argument:

```bash
mlx-one inspect Qwen/Qwen3-Reranker-0.6B
mlx-one rerank Qwen/Qwen3-Reranker-0.6B "What is unified memory?" \
  "CPU and GPU share one memory pool." "A recipe for sourdough." \
  --top-k 2 --max-length 512
```

## Settings and troubleshooting

- `score` is the yes-minus-no logit difference; `probability` is the yes probability
  after a two-logit softmax. Neither is a cosine similarity. Equal scores retain
  stable input ordering.
- The implemented profile has a configured 40960-token maximum; the service
  defaults to 8192. Start small and qualify memory separately.
- `top_k` limits returned results; it does not replace scoring the supplied pairs.
- A missing single-token answer or task mismatch requires the correct tokenizer
  and reranker bundle, not an embedding or chat checkpoint.
- Reranking does not expose server APC or block generation. Retrieval training
  and quality improvements need their own pinned evidence.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[retrieval service](../../../retrieval/service.py),
[backend-free tests](../../../../../tests/test_native_qwen3_retrieval.py), and
[synthetic Metal tests](../../../../../tests/test_native_qwen3_retrieval_mlx.py).

```bash
pytest -q tests/test_native_qwen3_retrieval.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen3_retrieval_mlx.py
```

See [qualification levels](../../../../../README.md#capability-levels) for the
distinction between tested contracts and measured task quality.
