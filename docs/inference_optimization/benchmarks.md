# Current Phase 1 repair measurements

Source and complete tables: [phase1_repair/results.md](../qualification/archived-evidence.md).
Canonical Qwen3-0.6B INT8, FP16 execution/KV, reference attention, actual B1:
seven exact 512/128 HTTP requests, five warm groups. TTFT p95
3782.7811ms; prefill median 148.6876 TPS;
decode median 17.1571 TPS; peak MLX 1211149036 bytes.
This bounded scope does not complete the twelve-request-group five-case baseline.

204 operator pairs were measured, 191 faster and 13 slower. Whole-model fused
performance is BLOCKED by numerical failure; there is no model speedup or
optimized-model memory comparison. Reference TTFT exceeds its declared budget.
Unexecuted full-reference/model B2/B4 and remaining checkpoint cases remain
HARDWARE_PENDING. The older benchmarking specification below is retained.

---

# Benchmark handoff

All new hardware measurements: **HARDWARE_PENDING**. Target: Apple M4, 32 GiB.
This session intentionally runs only lightweight tests/static checks. Installed
MLX is 0.32.2; sandboxed sysctl telemetry is unavailable. No native performance,
thermal condition or fused kernel success is inferred from package installation.

Phase 0 workloads per checkpoint: `(prompt, output, slots)` = `(512,128,1)`,
`(2048,128,1)`, `(4096,128,1)`, `(2048,128,2)`, `(2048,128,4)`.
The existing native runner supplies cold/warm separation, deterministic traffic,
five repetitions, stage observations, memory/RSS supervision and checksummed raw
results. B2/B4 reference_dense describes cooperative reference serving slots,
not a claim of fused batch execution. Model-only prepared-history decode and
end-to-end serving measurements remain distinct.
The new track verifies exact rendered input lengths and rejects early EOS as an
incomplete requested output workload. Such a failure is a workload/protocol
blocker to investigate, not a fabricated 128-token throughput result.

Phase 1 hardware commands:

```sh
scripts/qualification/fused-smoke.sh
scripts/qualification/fused-full.sh --kernel-only
scripts/qualification/fused-correctness.sh --checkpoints qwen3-small --checkpoint-paths /tmp/mlx-one-checkpoints.json
scripts/qualification/fused-performance.sh --checkpoints qwen3-small --checkpoint-paths /tmp/mlx-one-checkpoints.json --correctness-report /absolute/path/to/fused-results.json
```

Kernel-only commands exercise 223 identical-operand cases without loading weights.
Whole-model diagnostics use shared-attention checkpoints; an architecture outside
that dispatcher is explicitly UNSUPPORTED, never coerced into a dense layout.
The first failing numerical test stops expansion. Preserve its receipts to
diagnose the existing Qwen failure; do not proceed to performance on a failed
checkpoint. Kernel-only results cannot qualify whole-model execution. Kernel
timings report five samples, medians, memory and reference/fused latency ratios.
Whole-model optimized TTFT/prefill/decode comparisons remain a later Phase 1
acceptance requirement after numerical results are available.

Create an explicit local mapping, for example `/tmp/mlx-one-checkpoints.json`:

```json
{"qwen3-small": "/absolute/local/Qwen3-0.6B-8bit"}
```

Run from the repository:

```sh
scripts/qualification/smoke.sh
scripts/qualification/correctness.sh --checkpoints qwen3-small --checkpoint-paths /tmp/mlx-one-checkpoints.json
scripts/qualification/performance.sh --checkpoints qwen3-small --checkpoint-paths /tmp/mlx-one-checkpoints.json --correctness-report /absolute/path/to/correctness.json
scripts/qualification/full.sh --checkpoints qwen3-small --checkpoint-paths /tmp/mlx-one-checkpoints.json
```

Omit `--checkpoints` to retain all six P1 recipes; provide all six paths, including
verified native INT8 conversions for SmolLM2. No model download is initiated.
Default per-worker RSS ceiling is 20 GiB, capped at 24 GiB. The existing Metal
preflight may lower it to the recommended working set. Workers execute serially.
Optional `--include-paged` adds separate experimental paged correctness checks;
the baseline performance track remains reference_dense.

Each invocation prints its result directory beneath
`qualification/results/inference_optimization`. Preserve the entire directory,
including nested manifests, numerical receipts, logs, checksums, summaries and
RSS samples. Top-level `measurements.json`, `measurements.csv`, `handoff.json`
and `summary.md` index the observed outputs. A handoff PASS means the selected
commands succeeded, never production/performance promotion.

Historical performance artifacts under `qualification/results/p0_engine` and
`qualification/results/p01_math` remain available for audit. They do not form a
valid new-source speedup comparison. New baseline-versus-optimized numerical
speedups, TTFT, prefill TPS, decode TPS, batch throughput and peak memory are
pending comparable native runs; no values are fabricated.
