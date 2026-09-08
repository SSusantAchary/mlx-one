# Experimental text SFT vertical slice

This example exercises the implemented workflow without presenting an
unmeasured quality claim. The model revision is pinned; training and benchmarking
download approximately 3 GB of weights and require an Apple Silicon terminal
with Metal available.

Run the complete resumable workflow with:

```bash
mlx-one workflow text-sft \
  --config examples/text-sft/qwen2.5-coder-1.5b.workflow.yaml \
  --resume
```

The second-architecture candidate uses
`examples/text-sft/smollm2-1.7b.workflow.yaml`. Both remain candidates until the
generated evidence passes and is promoted through `mlx-one registry promote`.

```bash
mlx-one inspect Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a

mlx-one plan train \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --hardware m4-air-32gb --method lora

mlx-one data validate --dataset examples/text-sft/train.jsonl \
  --layout prompt-completion

mlx-one train \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --dataset examples/text-sft/train.jsonl \
  --config examples/text-sft/train.yaml

mlx-one evaluate --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --dataset examples/text-sft/eval.jsonl \
  --predictions examples/text-sft/predictions-smoke.json \
  --runs-dir runs --run-id scoring-smoke --json-output

mlx-one benchmark inference \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --prompts examples/text-sft/prompts.json --runs-dir runs
```

The supplied prediction file tests scoring mechanics only; it is not generated
by the model. To qualify this workflow, capture base and adapter predictions with
identical generation settings, evaluate both, compare them with
`examples/text-sft/quality-gate.yaml`, export the adapter, reload it, and record a
deterministic smoke prompt. Only then may a capability be promoted to quality or
hardware verified.
