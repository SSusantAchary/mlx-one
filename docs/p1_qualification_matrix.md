# P1 qualification matrix

Pins: `mlx_one.p1_profiles.CHECKPOINTS`. FP16 KV only in P1. Existing defaults are
unchanged. No capability is production-qualified by this delivery.

| Checkpoint | Implementation | CPU contracts | Checkpoint validation | Metal / performance |
|---|---|---|---|---|
| Qwen3-0.6B INT8 | dense + paged B1/B2/B4 | passed instrumented fixtures | pinned INT8 loaded; serial parity passed | dense B1 prefix FAIL; B2/B4 NOT_RUN; paged BLOCKED; performance pending |
| Qwen3-1.7B INT8 | same adapter + scaling profiles | shared contracts | metadata inspected; weights missing | pending |
| SmolLM2-360M | Llama adapter + native INT8 conversion | conversion integrity + batching passed | pinned base/converted weights missing | pending |
| SmolLM2-1.7B | same infrastructure + scaling profiles | shared contracts | pinned base/converted weights missing | pending |
| LFM2-350M INT8 | hybrid dense + attention-only paged | full-state, fork/resume, cancellation, prefix and reload fixtures passed | metadata inspected; weights missing | pending |
| LFM2-1.2B INT8 | same hybrid adapter + scaling profiles | shared contracts | metadata inspected; weights missing | pending |

CPU page comparisons use a differential oracle. They do not compile Metal or
validate actual INT8 checkpoint computation. Native tests must establish greedy
and logits parity, B>1 GPU forwards, isolation, skipped prefill and cleanup.

Performance gates: B4 aggregate decode throughput >=1.25x matched serial; optimized
B1 decode regression <=5%; B4 p95 model-token gaps <=1.5x B1, with five measured
repetitions and declared absolute TTFT/token-gap/fairness limits. An 8K shared prefix
also requires parity, measured skipped work and >=10% lower warm p50 TTFT against
a matched no-prefix run. Missing native evidence leaves auto at reference dense.

Earlier native worker attempts stopped with memory_watchdog_unavailable because sandbox
RSS access was unavailable. A separate import probe failed with
`[metal::load_device] No Metal device available`. Neither result is a native pass.
Generated failure/provenance artifacts and JSON/CSV/readable matrices stay under
`benchmarks/results/p1/<host>/<checkpoint>/<run>/` and are not committed.

## Native M4 cohort, 2026-10-14

Native host access now succeeds: GPU calculation, live child RSS and MLX memory
telemetry pass. The final source digest is
`ee65d6ed2ac3b58d184bc37d4bc7168a084b7bfbf35d8eb4fa48bf9413af5b5d`.
Qwen3-small dense B1 has exact forced continuation/KV and independent restored-cache
agreement, but fails full-prompt prefix logits tolerance (maximum 0.064453125).
Only that diagnostic scope was executed; all 2,592 optimized performance
configurations remain NOT_RUN. No B>1 GPU batching, paging, service targets,
throughput ratios, 8K prefix speedup, hybrid lifecycle or pressure stability has
been established. Other checkpoints remain NOT_RUN and their assets are missing.

See [native report](p1_native_hardware_report.md) and
[failure log](p1_native_failure_log.md) for scope, commands and artifact references.
