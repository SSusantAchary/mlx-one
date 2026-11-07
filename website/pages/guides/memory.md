# Inspect a checkpoint and plan memory

Inspect metadata before loading weights. A configured context limit is distinct
from measured memory capacity on your machine.

## Check your environment

```bash
mlx-one doctor
mlx-one hardware list
mlx-one hardware detect --json-output
```

## Inspect a model

```bash
mlx-one inspect Qwen/Qwen2.5-1.5B
```

Inspection is metadata-only. It does not run remote model code or open pickle
checkpoints. Use `--revision` for a pinned model commit and `--offline` to require
existing local or cached assets.

## Make a workload plan

```bash
mlx-one plan inference \
  --model Qwen/Qwen2.5-1.5B \
  --hardware m4-air-32gb \
  --precision bf16 \
  --context-length 2048
```

Plans estimate a particular model, precision and context on a hardware profile.
They are not measurements or guarantees of maximum context. Start with a bounded
workload and check the [recorded performance](../performance/index.md) and
[runtime limitations](../../../docs/experimental/known-limitations.md#runtime-limitations).

Dense reference execution and one active sequence remain the default. Advanced
cache and batching modes require their own capability and evidence matches;
see [runtime experiments](../../../docs/experimental/inference-runtime.md).
