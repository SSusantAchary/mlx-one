# Native LFM2 MoE and GPT-OSS implementation

Implemented in the actual local checkout on 2026-10-21. Entry revision:
`b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`. The worktree already contained
substantial tracked and untracked work; existing changes and frozen numerical
evidence were preserved. No commit, staging, publication or large weight download
was performed. Local references are `references_libs/mlx-lm-main` and
`references_libs/transformers-main`; their inspected file hashes are recorded in
[source-provenance.json](source-provenance.json).

Both target families are **CODE_IMPLEMENTED / SYNTHETIC_TESTED /
ARCHITECTURE_IMPLEMENTED**. Both exact large checkpoints are
**NOT_DOWNLOADED_NOT_SMOKE_TESTED / MANUAL_TEST_PENDING**. Neither is advertised
as `MODEL_SMOKE_TESTED`, `EXPERIMENTAL_SUPPORTED` or `USER_VALIDATED`.

## Native implementation and integration

- The existing `lfm2_moe` registration was repaired. Its earlier implementation
  used softmax, included expert bias in mixture weights and evaluated all
  experts. It now uses sigmoid scores, selection-only bias, unbiased gathered
  probabilities, optional top-k normalization with the reference epsilon,
  routed scaling and native selected-expert gather operations.
- LFM2 reuses the established reference convolution, Q/K-normalized attention
  and rotary math. Initial dense FFNs use the checkpoint's actual width without
  the dense LFM2 auto-adjustment. Existing hybrid cache/state adapters remain
  usable, including their previously tested sliding/ragged extensions.
- GPT-OSS is added to the existing lazy architecture registry, bringing it
  from 29 to 30 registrations. The existing native loader, cache planner,
  `Engine`, `ModelManager`, `GenerationEngine`, CLI and HTTP/SSE paths execute it.
  No second backend or registry was introduced. Standalone task-service adapter
  qualification and optimized paging/batching are not newly claimed for GPT-OSS.
- GPT-OSS uses explicit `head_dim`, biased attention projections, attention
  sinks, per-layer sliding/full masks and separate bounded/full KV stores.
  Sparse top-k routing and capped OpenAI SwiGLU preserve verified semantics.
  YaRN honors the checkpoint's `truncate=false`, unlike the older local MLX-LM
  helper that did not forward this setting. Transformers was inspected as an
  independent explanation of that flag and of both routers.
- Config parsers reject missing core fields, conflicting aliases, unknown
  architecture fields and invalid schedules/experts/RoPE/quantizers. Cache
  validation checks positions, layer counts, state types and shapes before
  ordinary layer state can mutate. Existing engine-owned row wrappers retain
  their own row/mask validation.
- Checkpoint parameter IO is completed on the loading thread. GPT-OSS's small
  non-parameter YaRN frequency buffer is also materialized before a bundle can
  move to the device executor. This fixes the first-use cross-thread stream
  failures found with quantized FP16/BF16 fixtures; callers need no warmup/eval.
- The mixed-quantization loader preserves complete per-module maps. It checks
  module names, packed tree shapes, uint32 payloads and floating affine scale/
  bias dtypes, including stacked experts. Native MXFP4 uses E8M0 uint8 scales and
  has no affine quantization bias. Original interleaved GPT-OSS `_blocks` and
  `_scales` layouts are sanitized and tested as well as the actual already
  sanitized artifact names.
- A real bounded MXFP4 `gather_qmm` probe detects incompatible cores with
  `UNSUPPORTED_QUANTIZATION: INCOMPATIBLE_MLX_CORE` and a paired MLX/Metal upgrade
  command. MXFP4 bytes are never interpreted as affine Q4. The installed
  `mlx==0.32.2` / `mlx-metal==0.32.2` pair passes the probe on Metal.
- Generation reads all EOS IDs from `generation_config.json`. Actual checkpoint
  Jinja templates render natively, including generation-span tags and optional
  builtin-tool defaults. Harmony supports the artifact's older token aliases
  and newer spellings, hides protocol/nonfinal/tool channels and renders final
  text in Python, CLI and HTTP/SSE. Model-token TTFT and first-visible-text TTFT
  have separate semantics. Accepted generated counts include nonfinal model
  tokens and exclude EOS, consistent with the existing generation contract.
- Valid local directories never require Hub access. Existing tokenizer,
  safetensors/index, cancellation, bounded FIFO, single-active serving and
  cleanup behavior remain in use. A detected Metal working-set lower bound
  rejects checkpoints larger than available working-set headroom; actual
  runtime/transient memory still depends on workload.

## Actual checkpoint metadata

Only configs, tokenizer metadata/templates, indexes and licenses were retrieved.
Pinned metadata fixtures retain the full maps and SHA256 provenance in
`tests/fixtures/moe_metadata/`.

| Artifact | Verified metadata |
|---|---|
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` @ `667408d494bde191c2cb9dd4f4549827816b9275` | 40 layers, hidden 2048, 32 Q / 8 KV heads; attention at 2, 6, …, 38; dense width **11776**, expert width **1536**, 64 experts/top-4, first two dense; conv length 3; theta **10000000**; tied embedding Q6/group64, affine Q4/group64 elsewhere, FP32 router; EOS 124900 |
| `mlx-community/gpt-oss-20b-MXFP4-Q4` @ `f356f2747216d7e98fee755df25987459fc19089` | 24 layers, hidden 2880, **head_dim 64**, 64 Q / 8 KV heads, 32 experts/top-4; sliding 128/full alternating; MXFP4/group32 default plus **122 module overrides**; Q4 attention/embedding/head and Q8 routers; YaRN factor32/original4096/theta150000/truncate=false; EOS **200002, 199999, 200012** |

Both configure 131072 positions. That is not a demonstrated practical context
on M4/32GB. Manual smoke should begin at 128–1024 total tokens.

## Verification and remaining work

Exact commands, outcomes, artifact hashes and links to retained logs are in
[verification.json](../qualification/archived-evidence.md). The default baseline before changes was
**1848 passed / 68 skipped / 3 warnings**. New architecture tests run automatically
when Metal exists; none require full model weights. They cover float32/FP16/BF16
forward and continuation, actual cache contents/reset/request isolation, invalid
states, synchronous batch forward, deterministic nonuniform MoE math, original
and native packed tensor layouts, mixed quantization, all EOS IDs, Harmony at
every character split, real native Python/CLI/HTTP/SSE, authentication,
cancellation/recovery and production import isolation.

| Final check | Actual result |
|---|---|
| Default suite | **1892 passed, 68 skipped, 3 warnings** |
| Full Metal-enabled suite | **2158 passed, 18 skipped, 3 warnings** |
| Focused new architectures plus candidate-loader regressions | **66 passed, 2 warnings** |
| Ruff debt gate | **PASS; 65 existing findings, zero new findings** |
| Wheel/sdist exact-source, UI, license and payload audit | **PASS** |
| Fresh offline wheel install and `pip check` | **PASS** |
| Installed GPT-2 Python/CLI/authenticated TCP HTTP/chat/SSE | **PASS** |

There are no unresolved final test failures. The skipped cases are existing
opt-in/environment scopes; all 43 new architecture/protocol cases ran on Metal.

The additional first-use quantized FP16/BF16 Engine checks deliberately run
before any caller-side forward/evaluation, preventing a warmup from hiding the
thread/stream defect. Final regression and packaging checks include this fix.

The clean build uses a temporary source copy, both wheel and sdist, a fresh venv
outside the checkout and cached dependency wheel bytes. The audit checks exact
production Python/UI source identity, license notices and absence of weights,
reference trees and private artifacts. Installed-wheel Python, CLI, authenticated
TCP HTTP completion/chat/SSE, malformed artifact recovery and cleanup use an
existing local GPT-2 checkpoint. No reference package is installed or imported.

Fresh bounded checks of existing cached source checkpoints are retained as
[GPT-2](../qualification/archived-evidence.md), [Qwen3.5](../qualification/archived-evidence.md) and
[Gemma3](../qualification/archived-evidence.md). Qwen3.5's stored-source short generation and
prefill/continuation/reset pass; its converted Q4 and previously broken paths
remain unqualified. Gemma3 remains eligible under the ordinary functional
policy, with its historical numerical differences and weak/repeated answers
disclosed. Those source smokes do not certify broader quality or context.

The previous ten checkpoint rows remain in the packaged support manifest. Nine
retain bounded experimental functional support based on the historical real
checkpoint/state/API proof; Qwen3.5 remains experimental and unqualified. The two
new targets are architecture-only. The [functional admission policy](../model-admission.md)
retires broad E/KL/NLL/FP64/partition research as default admission gates;
historical research reports and their original outcomes remain unchanged.
Administrative release/publication approval remains pending.

The owner still needs to download and actually load each large target, inspect
generated final answers and every interface, exercise cancellation/reset/reload,
and measure practical memory/context/TTFT/throughput on the intended hardware.
No performance or full-checkpoint success is inferred from synthetic tests.

## Changed files and owner commands

[changed-files.json](../qualification/archived-evidence.md) is the exact task-specific file list;
it excludes unrelated pre-existing worktree changes. Code adaptation retains
Apple MIT attribution and the packaged shared upstream license. Fixture
metadata/templates retain source licenses. Liquid weights use `lfm1.0` and
GPT-OSS's model card declares Apache-2.0; no weights are redistributed.

Use the [copy-ready manual inference runbook](manual-inference.md) for immutable
local directories, native CLI/Python/chat, authenticated loopback HTTP/SSE,
cancellation/reset/reload and structured memory/performance capture.
