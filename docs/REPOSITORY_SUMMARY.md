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
PR-03 through PR-10 are implemented as ordered experimental review boundaries.
Latest validation: 875 backend-free tests, 129 native tests and four pinned dense/
prefix regressions pass. **Pinned Qwen3-small INT8 paged B2 still fails FP16 logits
qualification**; greedy agreement does not override that failure. No production
or performance promotion follows from these changes.

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

PR-03 adds a bounded fair `DeviceExecutor` and single-flight `ModelPool` with
model epochs, leases, pinning, TTL/LRU eviction and draining unload. Seven CPU
ownership tests pass, including a red-before/green-after disposal/reload race.
Native multi-model stress remains pending. PR-04 connects these owners to serving.

PR-04 bridges existing Engine and HTTP execution to the shared device owner and
adds `InferenceService` with residency leases. Live server streams prevent model
unload; reference slots remain separate B=1 forwards. Public signatures, result
types and SSE framing are preserved. 59 affected tests pass with two inherited
warnings. Native cross-model service stress remains pending.

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

PR-05 adds authoritative model-provided per-layer execution layouts, per-layer
dense packing and independent K/V widths. Evaluated component ownership swaps
roll back all layers and reservation counters if reconciliation fails. Cross-KV
tracks encoder identity/length separately from decoder position; generic recurrent
math remains unavailable. 93 affected CPU tests pass. Native expanded-layout
qualification remains pending.

PR-06 registers native GPT-2/Llama/Qwen2/Qwen3 causal adapters, validated work
packing/result mappings and per-layer head descriptions. GPT-2 learned positions
use real row positions rather than padded extent; unsupported paging fails early.
109 affected CPU tests and 12 tiny native B1/B2/B4 cases plus one legal-context
boundary passed. Other sizes/checkpoints and performance remain unqualified.

PR-07 adds OpenELM varying-layer and Qwen2-MoE adapters, explicit mask/position
handling, padding-aware router auxiliary statistics and conservative all-expert
scratch forecasts. 105 affected CPU tests and six tiny FP16 native B1/B2/B4
cases pass, including per-layer KV and real-token router parity. Native forward
traces and allocator/RSS observations stay local; no large-checkpoint or
performance qualification is inferred.

PR-08 binds LFM2/LFM2-MoE to component-aware hybrid execution. Ragged rows
retain private carries and absolute sliding positions; adoption uses the shared
transaction. Native testing exposed router scratch entering MLX parameters;
private storage now preserves immutable weight metadata and inspection properties.
144 affected CPU tests, 13 native hybrid/legacy tests and eight final greedy/state
checks pass. Prefix hits skip 16/32 real tokens with namespace isolation. Real
checkpoint Gate E and performance remain pending; new MoE paging stays disabled.

PR-09 integrates text-only Qwen3.5 compound convolution/gated-delta state. Equal-
offset/width buckets prevent padding from advancing recurrence; FP16 convolution
and FP32 recurrent tensors are cloned and atomically committed together. Versioned
v4 snapshots reject missing tensors, incompatible transitions and dtypes, and
failed writes preserve the previous generation. 134 affected CPU tests and eight
tiny native/legacy tests pass, including B1/B2/B4, evaluated cancellation, prefix
restoration, serialization and divergent continuation. Real Qwen3.5 checkpoint,
VLM serving, MTP and performance qualification remain pending or later milestones.

PR-10 expands direct segmented paging through the same per-layer adapters for
GPT-2, Llama, Qwen2/3, OpenELM, Qwen2-MoE, LFM2/MoE and Qwen3.5 text attention.
Convolution/recurrent tensors remain private; sliding-window paging is explicitly
rejected. Existing kernels/layouts are reused. Layout/position validation and
reversible multi-row adoption prevent partial reconciliation; cleanup attempts all
staged releases. Forecasts and diagnostics include both compound tensors. The
lifecycle/parity harness uses the scheduler's state buckets and compares complete
compound state, including serialized resume/fork/cancellation/reload.

Native checkpoint tests exposed padded INT8 attention changing reduction widths.
Compact real-row reductions repair dense B2/B4 while retaining one batched model
call. Floating execution policy v4 invalidates older arithmetic evidence; omitted
policy dispatch remains unchanged. Final pinned Qwen3-small dense B1/B2/B4 and
fresh-full-prefill prefix checks pass with zero measured logits error. The paged
B2/page-16 checkpoint comparison remains **failed** at unchanged FP16 tolerances
(maximum logits error 0.033203125; greedy agreement 1.0). It has no repaired or
qualified replacement; further paged checkpoint expansion is stopped.

Final validation: **875 passed, 28 skipped**, two inherited warnings; **129 native
regressions** and **four pinned dense/prefix tests** pass under Metal/RSS preflight
and the 21.33 GiB watchdog. Ruff retains exactly **65 inherited findings**, with
zero new findings. The 28 opt-in skips in the backend-free run are not native
passes. Other real checkpoints, multi-model stress, paged INT8 correctness and
performance/service gates remain unqualified. All 55 pre-existing local file
states are preserved. Generated runs, failures, source/test checksums and CPU
logs remain local under `benchmarks/results/p1/<host>/`; none are committed.

PR-11 adds bounded retrieval task microbatches to the inference actor and residency
service. BERT/MPNet/Qwen3 vectors, native yes/no reranking and ragged LFM2-ColBERT
features retain original input order. Native-task contracts keep encoder work
separate from decode KV; cancellation/deadlines release residency leases. BERT
learned positions require right padding (reproduced and fixed in the existing API);
reranker loading now returns its canonical registry identity. 71 affected CPU
tests and 19 tiny native/legacy tests pass. Checkpoint quality, cross-task service
performance and the earlier paged INT8 failure remain unqualified.

### PR-12 — native vision task bridge

Qwen2-VL, Qwen2.5-VL, Qwen3.5 and explicitly native-format LFM2-VL
now have static-image task adapters. Qwen3.5 retains its separate text adapter.
Strict processor/layout metadata, bounded private media leases and evaluated
complete-state decoding preserve multimodal RoPE offsets and hybrid carries.
Decoders remain cooperative singleton forwards; no VLM GPU batching claim.
Publisher-format LFM assets without a native conversion contract fail closed.
56 focused CPU tests, 12 supervised tiny native tests and the full backend-free
suite (891 passed, 30 skipped, two inherited warnings) pass. The first native
fixture failure is retained locally; its continuation mask was corrected.
Real checkpoint processor/quality/performance gates remain pending.
