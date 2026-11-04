# Original benchmark reproduction

The owner report and all files under
`local/competitor-comparison/runs/comparison-20261101/` remain hash-identical to
the initial audit. No weights, tokenizer assets, model revisions or prompts were
downloaded, overwritten or converted. The original benchmark/setup scripts and
owner report were not edited.

The original script was executed unchanged with its existing isolated
environments, a new output path and a new report path. All 27 process sessions
passed. Each used one excluded warmup plus two timed raw 128-token prompts, greedy
generation, 128-token maximum, natural EOS, SSE, and disabled cross-request cache.
All actual checkpoint assets were rehashed by the runner. Preparation was not
rerun for this reproduction; retained pinned prompt IDs/hashes were reused.

| Model | Frozen historical MLX-One E2E tok/s | Reproduced E2E tok/s |
| --- | ---: | ---: |
| Qwen2.5 Q8 | 21.23 | 20.81 |
| Qwen3 Q8 | 20.13 | 19.99 |
| SmolLM2 BF16 | 14.75 | 14.65 |

The first two historical values have six fixed-length samples. The historical
SmolLM2 14.75 value is its five-prompt shared full-length view; the reproduced
14.65 is the unfiltered native view (all native completions were 128). Its EOS
case remains in all unfiltered cross-stack data; consult `benchmark-comparison.md`
for separately labeled shared subsets. No p95 is interpreted as an SLA.

Raw reproduction: `local/mlxboost/runs/baseline-original-20261101/results.json`.
Captured startup commands, readiness, server exit logs, per-request SSE events,
usage, model/prompt hashes and environment inventories are retained. The original
benchmark strips PYTHONPATH and launches the installed wheel; subsequent
experiments use an explicit local-source launcher and are labeled accordingly.

MLX-One remains MLX/Metal 0.32.2; MLX-LM/oMLX remain 0.32.4 with their original
distinct MLX-LM dependencies. These are original whole-stack protocol results,
not matched-dependency measurements. The separately pinned-source numerical
diagnostic uses MLX 0.32.2 for both implementations.

Benchmark-owned servers never overlap; native and HTTP campaigns are serialized.
Background host activity, CPU/GPU temperature and power were not controlled or
sampled. Their contribution to between-run variation is unresolved.

Reproduction command:

```sh
local/competitor-comparison/venvs/client/bin/python scripts/compare_competators.py run \
  --output local/mlxboost/runs/NEW-BASELINE \
  --report local/mlxboost/runs/NEW-BASELINE/report.md
```

The original setup script was inspected and not run: existing environments already
passed dependency checks. This avoided upgrading the development environment.
