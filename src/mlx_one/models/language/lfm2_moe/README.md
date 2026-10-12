# LFM2-MoE

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `lfm2_moe`. The implementation combines LFM2's convolution/attention
decoder with top-k expert routing, optional expert bias, and router outputs.
Configuration, expert stacking, hybrid caches, and synthetic execution have tests.
A native text loader is registered, but exact-checkpoint integration is unqualified.

[LiquidAI/LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B) is the upstream
family reference. Its total-parameter footprint places it outside the initial
≤3B qualification tier, despite the smaller active-parameter count.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Token IDs → token embeddings
    ↓
┌──────────────────────── Hybrid decoder ────────────────────────┐
│ RMSNorm → short convolution OR GQA attention → residual        │
│                    convolution state / attention KV            │
│ RMSNorm → dense gated MLP (initial configured layers)          │
│        OR router → top-k expert MLPs → weighted sum            │
│ Feed-forward output → residual                                 │
└────────────────────────────────────────────────────────────────┘
    ↓
Final RMSNorm → LM head → logits (+ optional router outputs)
```

## Papers and upstream references

Backbone paper: [LFM2 Technical Report](https://arxiv.org/abs/2511.23404).
Variant reference: [LiquidAI/LFM2-8B-A1B model card](https://huggingface.co/LiquidAI/LFM2-8B-A1B).

The local MoE path uses configured dense/expert layers; the backbone paper is
not a separate qualification claim for this MoE checkpoint.

## Installation and assets

Use Python 3.10+ and `python -m pip install -e .` from the repository root.
Execution needs Apple Silicon/Metal. A local experimental bundle needs compatible
`model_type=lfm2_moe` configuration, complete expert/convolution/attention
safetensors weights/index, `tokenizer.json`, tokenizer metadata, and chat template.
Keep dense-layer counts and expert settings consistent with the weights.

## Inspection and experimental chat

```bash
mlx-one inspect LiquidAI/LFM2-8B-A1B
```

Use this example only with a prepared local checkpoint that passes the native
contract and fits the host. Replace the path; this is not a qualified publisher load.

```python
from mlx_one import TextGenerationOptions, load_text_model, stream_chat

bundle = load_text_model("/path/to/lfm2-moe", offline=True)
for chunk in stream_chat(
    bundle,
    [{"role": "user", "content": "Explain mixture-of-experts routing."}],
    options=TextGenerationOptions(max_tokens=32),
):
    print(chunk.text, end="", flush=True)
```

Use the loaded checkpoint template. Raw `mlx-one generate MODEL PROMPT` performs
completion and does not automatically build a conversation.

## Cache, limits, and troubleshooting

The ordinary dense backend retains both attention KV and convolution state.
Block KV and hybrid prefix restore are unsupported. A request cannot safely
reuse attention alone while discarding convolution history.

Memory planning must include all experts, not only those active per token.
Check `num_experts`, `num_experts_per_tok`, dense-layer counts, and convolution
layout when a tensor contract fails. Do not discard unknown or missing experts.
Synthetic cache equivalence does not establish long-context, parallel, hardware,
or quality qualification for this family.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[architecture tests](../../../../../tests/test_native_lfm_architectures.py), and
[synthetic Metal tests](../../../../../tests/test_native_lfm_mlx.py).

```bash
pytest -q tests/test_native_lfm_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_lfm_mlx.py
```

The [family backlog](../../../../../model_list.txt) records the initial-tier
exclusion; the [cache matrix](../../../../../docs/cache-runtime.md) records hybrid restrictions.
