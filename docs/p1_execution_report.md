# P1 execution report

Base: `6d2c9a1bbc36f1972b468ca841977e3b342fbb36`.
PR-00: reference repositories inspected at immutable pins; P0 CPU suite refreshed:
704 passed, 23 skipped, two dependency warnings. 65 pre-existing Ruff findings.
No checkpoint weights downloaded or native P1 model run at this boundary.
Metal was unavailable in the P0 sandbox. Hardware validation remains pending.

Subsequent boundaries append actual changes, commands, failures and validation.
Generated artifacts belong under benchmarks/results/p1/<host>/<checkpoint>/<run>/
and remain uncommitted. No push or capability promotion is authorized by readiness.

PR-01: affine INT8 loading/conversion and immutable native adapter specification added.
Explicit FP16 conversion preserves packed uint32 payloads. Existing 4-bit defaults
remain. Affected suite: 50 passed; native loading/model parity pending.

## PR-02: Qwen3 dense continuous batching

The dense runner now uses the immutable native adapter for eligibility, head layouts and forward execution. CPU instrumented INT8 Qwen3 fixtures cover B1/B2/B4, unequal histories, serial token agreement and handle cleanup. Affected tests: **54 passed** (two existing dependency warnings). Changed-file Ruff passes. Real checkpoint logits and GPU batching remain pending.

## PR-03: chunked scheduling profiles

P1 profiles cover chunks 128/256/512/1024 with five measured repetitions and unchanged budget defaults. A deterministic 8K prefill fixture proves concurrent decoders receive service each iteration within the combined budget. Existing admission, reservation, cancellation and backpressure tests remain in the affected suite: **40 passed**. Native service-limit measurements remain pending.
