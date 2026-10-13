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
