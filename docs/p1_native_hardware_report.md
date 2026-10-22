# P1 native hardware qualification — 2026-10-14

**Status: incomplete; stopped at the Qwen3-small dense correctness gate.**
No production capability was promoted, no performance cohort ran, and no default
or public generation/HTTP/SSE interface changed.

Changes are in `mlx-one` on `qualification/p1-native-m4`, based on
`b698f88afc1729cd7d2dea53fd5a69f11b60246e`. Original dirty README/UI files and
other worktrees were preserved. Source digest for the final native cohort:
`ee65d6ed2ac3b58d184bc37d4bc7168a084b7bfbf35d8eb4fa48bf9413af5b5d`.

## Native preflight and assets

Apple M4, 32 GiB RAM, macOS 15.7.9, Python 3.12.11, MLX 0.32.2. GPU execution
returned 1240 after evaluation and synchronization. RSS and MLX active/cached/peak
telemetry were available. The final preflight recorded 100,918,853,632 free disk
bytes. The device working-set recommendation lowers the watchdog ceiling from
24 GiB to **22,906,503,168 bytes (21.33 GiB)**. No watchdog was disabled.

Only pinned `mlx-community/Qwen3-0.6B-8bit` at
`11de96878523501bcaa86104e3c186de07ff9068` was downloaded. Its 197 packed weight
headers passed INT8/group-64 layout inspection. Native loading validates affine
modules; explicit FP16 conversion preserves packed uint32 payloads. Checkpoint,
tokenizer and configuration checksums are in worker artifacts. The other five
assets remain missing; they were not downloaded after the correctness failure.

## Observed results

| Check | Result | Scope |
|---|---|---|
| GPU execution, child RSS watchdog, allocator telemetry | PASS | Native M4 preflight |
| Independent serial forced-token parity | PASS | 128-token target prompt, 32 forced steps |
| Tiny native Qwen suite | PASS | 5 tests; not checkpoint/performance qualification |
| Dense B1 forced continuation | PASS subcheck | 33 logits and 1,848 KV comparisons, maximum error 0 |
| Restored target versus independently restored native cache | PASS subcheck | Identical and divergent suffixes, maximum error 0 |
| Restored target versus fresh full-prompt oracle | FAIL | FP16 tolerances unchanged; maximum error 0.064453125 |
| Dense B2/B4, paged kernels, service/lifecycle/stress checks | NOT_RUN / BLOCKED | Correctness stop rule |
| Other checkpoints and all performance gates | NOT_RUN | Ordered expansion stopped |

Restoration genuinely reused 352/353 prompt tokens and left a suffix to execute.
Both prefix comparisons retained greedy agreement, but failed elementwise
`atol=rtol=1e-2`. Independent single-request restored execution shows the same
full-prompt discrepancy. This narrows the remaining investigation to native
full-prefill versus cached-suffix arithmetic; it does not waive the required gate.
Cleanup reported zero active handles, reservations, and allocated cache bytes.
The failed checkpoint worker's sampled peak RSS was 897,974,272 bytes; this is a
diagnostic process measurement, not a performance or peak-memory qualification.

## Repairs and harness changes

Removed a shadowing local import that prevented optimized parity evidence from
being written. Explicit floating execution now promotes attention scores before
softmax even without a padding mask; neutral masks no longer change arithmetic.
This does not create FP32 copies of KV histories or change omitted-policy execution.
Native red-before/green-after tests are linked in the failure log. The floating
policy is version 2; old/missing policy identities cannot qualify new execution.

The staged driver enumerates 2,592 optimized legal configurations and deduplicated
serial/cooperative dependencies. It records literal SmolLM2 8192-prompt rejections,
uses labeled maximum-legal substitutes, and validates checksummed correctness
artifacts against source, test source, hardware, logs, parity and cleanup before
performance. Comparison workers receive identical twelve-request traces and
explicit queue/budget settings. Internal token observers capture native IDs
without HTTP payload changes. Prepared-history forced-token forward throughput
excludes prefill/first-token work and remains separate from HTTP service throughput.
This new measurement tooling has CPU coverage; its native performance runs remain
unexecuted. Service, pressure and lifecycle profiles also remain unexecuted.

## Reproduction and artifacts

```sh
PYTHONPATH=src .venv/bin/python benchmarks/p1_native_qualification.py manifest
PYTHONPATH=src:.:tests .venv/bin/python benchmarks/p1_native_qualification.py correctness \
  --checkpoints qwen3-small
```

Run the native command outside the restricted sandbox, retaining supervision.
Do not run `performance` until the saved correctness report passes on the same
source and hardware. Each worker stays offline; downloads are explicit.

Final cohort metadata:
`benchmarks/results/p1/Susants-MacBook-Air.local/qualification/1791945161706526000/`.
Failed native checkpoint artifacts:
`benchmarks/results/p1/Susants-MacBook-Air.local/qwen3-small/1791945161706526000/dense/`.
Frozen serial oracle:
`benchmarks/results/p1/Susants-MacBook-Air.local/qwen3-small/serial-frozen-1791945215783446000/`.
Artifacts, generated JSON/CSV matrices and failures remain uncommitted.

Backend-free validation: **773 passed, 24 skipped**, two inherited dependency
warnings. Ruff: **65 inherited findings**, no new findings; no automatic rewrites.
Diff checks passed. Native skips were not counted as passes. Full 2,592-configuration
measurement, Metal paging, other families, throughput/service/prefix benefit,
pressure recovery and lifecycle qualification remain pending.

See [failure log](p1_native_failure_log.md) and [matrix](p1_qualification_matrix.md).
