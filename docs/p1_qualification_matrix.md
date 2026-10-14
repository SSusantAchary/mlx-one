# P1 qualification matrix

Pins: `mlx_one.p1_profiles.CHECKPOINTS`. FP16 KV only in P1. Existing defaults are
unchanged. No capability is production-qualified by this delivery.

| Checkpoint | Implementation | CPU contracts | Checkpoint validation | Metal / performance |
|---|---|---|---|---|
| Qwen3-0.6B INT8 | dense + paged B1/B2/B4 | passed instrumented fixtures | metadata inspected; weights missing | pending |
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

A native worker attempt stopped with memory_watchdog_unavailable because sandbox
RSS access was unavailable. A separate import probe failed with
`[metal::load_device] No Metal device available`. Neither result is a native pass.
Generated failure/provenance artifacts and JSON/CSV/readable matrices stay under
`benchmarks/results/p1/<host>/<checkpoint>/<run>/` and are not committed.
