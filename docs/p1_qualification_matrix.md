# P1 qualification matrix

Pins are authoritative in `mlx_one.p1_profiles.CHECKPOINTS`. FP16 KV only.

| Model | Asset status at baseline | Dense B1/B2/B4 | Paged B1/B2/B4 | Native qualification |
|---|---|---|---|---|
| Qwen3-0.6B INT8 | Public revision/config inspected | pending | pending | pending |
| Qwen3-1.7B INT8 | Public revision/config inspected | pending | pending | pending |
| SmolLM2-360M | Pinned official base; conversion required | pending | pending | pending |
| SmolLM2-1.7B | Pinned official base; conversion required | pending | pending | pending |
| LFM2-350M INT8 | Public revision/config inspected | pending hybrid | pending attention-only | pending |
| LFM2-1.2B INT8 | Public revision/config inspected | pending hybrid | pending attention-only | pending |

Promotion requires same-checkpoint parity, real GPU batch traces, state isolation,
cleanup and declared service budgets. Performance gates: B4 throughput >=1.25x
matched serial; B1 decode regression <=5%; B4 p95 model-token gaps <=1.5x B1.
The 8K prefix workload additionally requires actual skipped work and >=10% lower
warm p50 TTFT. No measurements are supplied by this table.
