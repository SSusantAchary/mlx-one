# Focused MoE combine port — 2026-11-17

**Operator: REJECT. Model qualification: BLOCKED. Production activation: NONE.**
Stock MLX remains selected. G03 remains **UNQUALIFIED** and the existing stock
cached/fresh parity gate remains **OPEN**. No model, engine, packaging, public
API, Phase 0 or G03 source was changed; no full regression suite was run.

No usable local MoE checkpoint was found in the default Hugging Face cache,
MLX-One calibration cache or local checkpoint records/configs. GPT-OSS and
LFM2-MoE fixtures contain metadata, not pretrained weights. No model was fetched;
dense models and fabricated weights are not presented as checkpoint evidence.

## Code and eligibility

Only `benchmarks/moe_port/combine.py`, `pilot.py`, `NOTICE.txt` and
`tests/moe_port/test_combine.py` were added, plus this report. The isolated
per-instance `CombinePrototype(enabled=True)` uses `mx.fast.metal_kernel` and
Phase 0 configuration/decision, profiling and immutable-record helpers. It is
excluded from product wheels and has no model registration or production flag.
Default construction and closed instances take the exact stock expression.

The reference is LFM2's same-dtype `sum(selected * scores[..., None], axis=-2)`:
K4, B1, FP16/BF16, width 1–4096, measured M4/MLX 0.32.2 only. Rank-3/rank-4
singleton leading axes and actual strides are handled without implicit copies.
Larger batches/sequences, other K/dtypes/backends/streams fall back. Pending
markers are bounded; context exit/exception closes the instance without touching
any original module. M2/M3 execution is not qualified and falls back.

Routing was not ported. Local LFM2 uses sigmoid, optional FP32 expert-bias
selection, argpartition order, unbiased gathered scores, optional normalization
with `1e-6`, scaling and a cast to activation dtype. GPT-OSS instead partitions
logits then applies precise selected-logit softmax. Upstream sorted top-k/tie
behavior is not a universal substitute. Router selection/weights stay unchanged.

## Actual focused results

Apple M4 Air/32 GiB; MLX/Metal 0.32.2. Synthetic `[1,4,2048]` FP16 expert outputs,
identical materialized inputs, three paired interleaved warm trials. Timings are
synchronized wall time including host submission/completion, **not GPU-only**.

| Corrected implementation | Stock MLX | Native |
| --- | ---: | ---: |
| Median latency | 0.536458 ms | 0.540542 ms |
| Extra peak active MLX allocation | 20,480 B | 4,100 B |
| Output size | 4,096 B | 4,096 B |

Paired stock/native ratios: **0.98115, 0.96578, 1.08677**. The >=10% repeatable
operator gate fails. Pilot output is bit-identical, maximum absolute error **0**.
There are **9 GPU-written execution markers**: 3 measured, 4 first/warmup,
1 correctness and 1 memory probe. Distinct evaluated slices prevent repeated-input
CSE across timed calls; marker reads occur after evaluation outside timing.
The marker costs 4 bytes. This is an operator memory probe with warm allocator
state, not a model-memory reduction or physical 8/16/24 GiB qualification.

**17 focused tests passed**: FP16/BF16, tails, cancellation, strided/broadcast/
reversed inputs, tied stock routing/bias/normalization, output shape/dtype,
default-off, larger-M/K/dtype/CPU/backend fallback and exception/close lifecycle.
Existing Phase 0 native import plus evaluated AXPBY smoke passed; existing
packaging APIs are unchanged, so no new wheel build or broad suite was needed.

The first prototype incorrectly assumed FP32 accumulation. Its pilot passed
random-input policy but cancellation tests failed in both dtypes (15 pass/2 fail).
The corrected implementation rounds each product **and addition** to T, matching
ZMLX's exact variant and MLX 0.32.2's small-column reduction. The same tests and
single three-pair geometry were rerun after this correctness fix, with unchanged
thresholds. Initial failed evidence is retained; no tuning followed the final
negative pilot.

Model decode tokens/sec, TTFT, prefill, logits/state/greedy parity and model peak
memory: **NOT RUN — MODEL BLOCKED**. The historical reference parity failure is
an independent promotion blocker. There is no end-to-end acceleration claim.

## Attribution and reproduction

Portions were adapted from **ZMLX**, by Hunter Bown and contributors (MIT), at
`d8cd0d88d4299ca6821d706f9d5b4a3520d688f5`:
[`src/zmlx/kernels/moe.py`](https://github.com/Hmbown/ZMLX/blob/d8cd0d88d4299ca6821d706f9d5b4a3520d688f5/src/zmlx/kernels/moe.py#L1172).
The exact-combine loop and no-FMA approach were adapted for B1/K4, strides,
tails, bounded GPU markers and fallback. The pinned patch guards, exclusions,
KERNELS and TOUR were inspected at the same revision; hashes are in the audit.
Full **copyright 2026 Hunter Bown and MIT permission/warranty text** accompany
[the prototype](../../benchmarks/moe_port/NOTICE.txt). Preserve that notice if
redistributing it. No upstream runtime dependency was installed.

Raw evidence: corrected pilot (local-only: `local/mlxboost-moe/20261117-1794936250077786000/pilot-corrected/9fa98070bc6a4fdb4d197632ccdfc649dc51bc0922b98f68dd1c0ab02ab72086.json`),
focused tests (local-only: `local/mlxboost-moe/20261117-1794936250077786000/focused-tests-corrected.log`),
audit (local-only: `local/mlxboost-moe/20261117-1794936250077786000/audit.json`),
final verification (local-only: `local/mlxboost-moe/20261117-1794936250077786000/final.json`),
seal (local-only: `local/mlxboost-moe/20261117-1794936250077786000/manifest.sha256`).
Original preflight, failed tests, initial source and both immutable pilots remain.

```bash
MLXBOOST_MOE_GPU_TESTS=1 .venv/bin/python -m pytest tests/moe_port -q --no-cov
.venv/bin/python benchmarks/moe_port/pilot.py \
  --output local/mlxboost-moe/new-replay --execute-gpu
```

Normal GPU approval applies. Rollback: leave `enabled=False` (the default), close
the operation wrapper, or remove these benchmark/test files. No model patch,
registry approval, dependency or installation change needs undoing.
