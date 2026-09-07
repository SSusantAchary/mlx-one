# M4 text calibration

`m4-air-32gb-text-v1` measures memory and performance for one controlled model
family on the maintainer's 32 GiB M4 MacBook Air. A result applies only to the
exact model revision, MLX/MLX-LM versions, workload, settings, and hardware stored
in its record.

## Matrix

The matrix uses immutable revisions of Qwen2.5 0.5B, 1.5B, and 3B. For every
model and context length (512, 1024, 2048, and 4096), it runs:

- BF16 and group-size-64 4-bit inference, with one cold and three warm generations.
- BF16 LoRA and 4-bit QLoRA, with one warm-up and five measured optimizer steps.

Inference generates 32 tokens. Training uses batch size 1, rank 8, scale 20,
dropout 0, at most the final 16 layers, AdamW, seed 42, and no gradient
checkpointing. Deterministic synthetic token sequences isolate runtime behavior;
the matrix does not assess model quality.

## Safety and evidence

The reference profile reserves 8 GiB for macOS and limits the MLX process to at
most 24 GiB or the device-recommended working set, whichever is lower. A cell is
skipped when its analytic lower bound already exceeds that budget. A running cell
is terminated and marked invalid if free memory falls below 5% or swap grows by
more than 512 MiB.

Each cell runs in a fresh subprocess and writes one atomic, versioned JSON record.
Records contain Metal peak memory, process RSS, swap delta, cold/warm timings,
throughput, thermal state, and software provenance. They exclude weights,
adapters, generated text, console logs, usernames, device identifiers, and local
artifact paths.

Thermal state is recorded as `unknown` when macOS does not expose it through
`pmset`; absence of a readable signal is never treated as a nominal reading.

## Running and resuming

First inspect the work without downloading weights:

```bash
mlx-one calibrate text --matrix m4-air-32gb-text-v1 \
  --output calibration-results/ --dry-run
```

Then run from an ordinary local Terminal session where `mlx-one doctor` reports
Metal as available:

```bash
mlx-one calibrate text --matrix m4-air-32gb-text-v1 \
  --output calibration-results/ --resume
```

Use `--select REGEX` to run a subset. `--resume` reuses terminal records;
`--resume --retry` reruns failed, invalid, or interrupted records. Converted model
artifacts live under `~/.cache/mlx-one/calibration-models` by default and must not
be committed.

The initial maintainer calibration stops at 3B to keep runtime and storage
practical on the reference machine. This is a validation boundary, not a model
size restriction in the planner or schema.

Published evidence must contain every safe matrix cell, structured reasons for
every skip or failure, `manifest.json`, and `summary.json`. Do not label a matrix
as measured when it contains dry-run output or estimated values only.

## Initial reference outcome

The first M4 32 GiB run produced 48 records: 44 completed, two BF16 LoRA
4,096-token cells were invalidated after excessive swap, and the 3B LoRA and QLoRA
4,096-token workers failed. All 24 inference cells completed. Leave-one-context-out
validation reported 5.98% median relative error, 17.85% p90 relative error, and
95.45% upper-bound coverage. The raw normalized records and checksummed manifest
are stored in `calibration/results/m4-air-32gb-text-v1`.
