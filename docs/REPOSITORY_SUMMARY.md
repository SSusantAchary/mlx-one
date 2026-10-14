# mlx-one — compact repository summary

Updated: 2026-10-15. Native MLX tooling for Apple Silicon; Python 3.10+, Apache-2.0.

## Repository and runtime

- Native model implementations and shared layers: `src/mlx_one/models/`.
- Python `Engine`, generation, streaming and lifecycle: `src/mlx_one/engine/`.
- HTTP/OpenAI-compatible serving, SSE and cooperative slots: `src/mlx_one/server/`.
- Text, vision, retrieval, audio, training, conversion and evaluation modules live
  under `src/mlx_one/`; support and qualification vary by checkpoint/task.
- Tests: `tests/`. Offline benchmark workers and qualification entry points:
  `benchmarks/`. Reference engines are not inference backends.

P0/P1 are already merged at `b698f88afc1729cd7d2dea53fd5a69f11b60246e`.
The native qualification follow-up is committed at `85d522c`.
Current development branch: `feature/registry-native-inference`, based on
`85d522ca0aa428c006ae939465d14301a9b0a4c2`.
The runtime reuses native adapters, `InferenceActor`, `BatchScheduler`, dense/paged/
hybrid runners, transactional cache ownership, prefix indexes, segmented arenas,
memory reservations and checksummed evidence. Optimized modes remain gated;
omitted-mode behavior and public generation/HTTP/SSE interfaces are preserved.
`auto` requires matching qualification evidence, otherwise reference dense.

## PR-00: registry baseline and prefix-failure reproduction

The approved platform roadmap covers all 27 registrations in PR-00 through PR-20,
delivered in strict numeric order. It includes the planned task integrations,
checkpoint-supported LFM2 Audio tasks, native MTP and external greedy drafts.
`sam3_tracker_video` may remain a declared nested component. These are future
deliverables, not new capability claims.

PR-00 adds two stages to the existing qualification entry point:

```sh
# Registry/source/dependency/asset/host inventory; never initializes Metal.
PYTHONPATH=src python benchmarks/p1_native_qualification.py inventory

# Offline pinned Qwen3-small, explicit FP16 floating execution and FP16 KV.
# Requires native Metal/RSS access; preflight and watchdog cannot be skipped.
PYTHONPATH=src python benchmarks/p1_native_qualification.py prefix-diagnostic
```

The inventory is generated from registry declarations. It records `lfm2_vl`,
`qwen2_5_vl` and `lfm2_audio` as architecture-only loader/task gaps, and records
`sam3_tracker_video` as a component of `sam3_video`. Qwen3.5's text-oriented loader
does not establish multimodal task availability. Declared loaders/capabilities
are separate from shared-service integration and qualification. Asset checksums
cover the six existing pinned P1 recipes; other task checkpoint assets remain
unassessed. Inventory and missing telemetry/asset limitations are checksummed.

| Check | Observed status |
|---|---|
| Backend-free suite | 798 passed, 24 skipped; two inherited dependency warnings |
| Affected CPU tests | 54 passed |
| Ruff | 65 inherited findings; zero new findings; changed files pass |
| Native M4 / 32 GiB preflight | GPU, RSS watchdog and MLX telemetry passed; ceiling 21.33 GiB |
| New opt-in native diagnostic test | 1 passed, 72 deselected; verifies reproduction, not qualification |
| Fresh full-prefill versus ordinary cached / restored / indexed prefix | **Failed** unchanged FP16 tolerance; initial maximum errors 0.02899169921875 / 0.064453125 |
| Independently restored versus prefix-index execution | Exact logits agreement; committed positions aligned |
| Cleanup | Zero live handles, reservations and allocated cache bytes; isolated snapshots disposed |
| Dense B2/B4, paging, other checkpoints, service/stress/performance | Not run or blocked |

The new diagnostic evaluates four independent paths on identical forced input IDs:
fresh full-prefill, ordinary cached continuation, independently restored native
state, and prefix-index restoration. Identical/divergent requests genuinely reuse
352/353 prompt tokens. Every tested logit is saved losslessly in safetensors;
results include greedy IDs, forced IDs, component positions/shapes/dtypes, state
comparisons at initial/final boundaries and cleanup. Missing assets/hardware stay
not-run; watchdog or worker failures cannot establish reproduction.

PR-00 reproduced the native failure; its historical qualification remains **failed**.
PR-00 changes observation/tooling only; it does not repair attention arithmetic,
change generation interfaces/defaults, or promote capabilities. Subsequent PR-01
contracts and PR-02 repair are recorded below. Historical P1 serial/continuation
passes retain their
original scope; the 2,592-configuration performance matrix remains unexecuted.

## Artifacts and delivery

PR-01 adds lazy, versioned runtime bindings for all 27 registrations, canonical
tasks/stages, task-neutral lifecycle with a legacy text mapping, explicit per-layer
descriptors, compatibility keys, task-cost work items and task-specific adapter
protocols. These are migration contracts, not implemented shared-service runners.
Metadata providers import no MLX; missing adapters fail explicitly. Unregistered
extensions cannot inherit a runtime binding. Registry ASR aliases normalize to
transcription; alignment and multi-vector tasks remain distinct.

PR-01 validation: 56 affected tests passed; full suite **824 passed, 24 skipped**
with the same two warnings. Ruff on `src tests benchmarks` retains exactly 65
baseline findings. A repository-root scan additionally reports 19 inherited
findings in unrelated local `proofcart_starter/`, which remains untouched.
Native PR-00 evidence retains its original source identity and failed status.
At PR-01, the fresh-full-prefill gate remained blocked on PR-02.

PR-02 isolates shape-dependent INT8 projection and attention reductions. Explicit
FP16 affine INT8 shared decoder arithmetic now uses existing MLX gathered
matrix-vector rows and FP32 attention accumulation. Activations/KV stay FP16 and
packed uint32 weights are preserved. Omitted-policy reference dispatch stays
unchanged. Execution policy is version 3, invalidating older arithmetic evidence;
extra FP32/gather scratch receives a conservative reservation and trace counter.
No new Metal kernels or external engines were introduced.

The supervised `prefix-parity` stage requires success against fresh full-prefill,
not merely reproduction. On M4/32 GiB under the 21.33 GiB watchdog ceiling, the
two original 353/356-token cases reused 352/353 tokens. All four paths over 32
forced steps had **zero measured logits/state error**, exact greedy agreement,
aligned positions and zero live handles/reservations after cleanup. Tolerances
remain `atol=rtol=1e-2`. Before the repair this regression test failed; after it,
the test passed on frozen source. This is narrowly scoped checkpoint correctness,
not performance, B2/B4, paged, other-checkpoint or public capability qualification.
Those gates and the 8K prefix TTFT target remain pending.

PR-02 validation: **104 affected tests**, **834 passed / 24 skipped** in the full
backend-free suite, the same two warnings, **7 tiny native tests** and **1 pinned
native regression test** passed. Scoped Ruff remains the same 65 findings with
zero new findings. Raw logits and before/after artifacts stay local.

```sh
PYTHONPATH=src python benchmarks/p1_native_qualification.py prefix-parity
```

Next ordered boundary: PR-03 device ownership, residency leases and single-flight
model loading. No later PR is included in these commits.

Checksummed inventory and run metadata stay local under
`benchmarks/results/p1/<host>/runtime-baseline/<run>/`; native logits, comparisons,
supervision and failure artifacts stay under
`benchmarks/results/p1/<host>/qwen3-small/<run>/prefix-diagnostic/`.
`--process-ceiling-bytes` can lower the watchdog ceiling, including GPU preflight.

PR-00/01/02 commits include only scoped contracts/runtime/diagnostic code, tests
and this summary.
Existing UI/document edits and generated artifacts remain local. No push or
capability promotion is authorized by implementation readiness.

```sh
git log -1 --oneline
git show --stat HEAD
git status --short
```
