# P1 native failure log — 2026-10-14

All paths below are relative to `benchmarks/results/p1/Susants-MacBook-Air.local/`.
Historical results retain their original source identity; they cannot qualify
the final code. No failed tolerance or service target was relaxed.

| Failure | Reproducer / evidence | Outcome |
|---|---|---|
| Optimized parity writer raised `UnboundLocalError: host_identity` | `qwen3-small/20261014T015819596899Z/native-baseline-v2-87c2df13d9c6691f/` | Removed local import shadowing module imports. Rerun `qwen3-small/20261014T020104347458Z/native-baseline-v2-87c2df13d9c6691f/` writes parity and correctly fails numerical acceptance. |
| Neutral padding mask changed explicit FP16 decode arithmetic | `qwen3-small/attention-before-1791944557674681000/` | Native regression failed before repair and passed afterward in `qwen3-small/attention-after-1791944626840822000/`. Scores now consistently enter FP32 softmax/value arithmetic. |
| Dense checkpoint prefix restoration differs from full-prompt oracle | `qwen3-small/1791945161706526000/dense/parity-qwen3-small-False-16-1.json` | **Unresolved.** Maximum errors 0.02899169921875 and 0.064453125; both fail FP16 elementwise tolerance. Independently restored single-request execution matches the target exactly and differs identically from the full oracle. |

The final dense failure occurred after exact agreement on all 33 forced-token
logits and 1,848 KV comparisons. Greedy agreement and successful cleanup do not
override the failed prefix logits gate. The next investigation must reproduce
full-prompt versus restored-suffix arithmetic without weakening the oracle.

Paged compilation/execution, Qwen3-large, both SmolLM2 conversions, both LFM2
checkpoints, concurrency/service, slow-consumer, pressure and performance cohorts
were not run after this failure. They are not passes or inferred regressions.
No push, capability promotion, tolerance change or default backend change occurred.
