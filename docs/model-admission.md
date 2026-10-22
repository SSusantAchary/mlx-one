# Functional architecture and checkpoint admission

Effective 2026-10-21. This supersedes prior numerical admission proposals as
the default model support policy. Frozen reports, logs and their original
PASS/FAIL decisions remain historical evidence and must not be rewritten.

The default gate follows the published MLX-LM functional approach: correctly
parse the architecture, load the checkpoint and execute useful native model
operations through the supported interfaces. No production `mlx_lm` or
`transformers` dependency is allowed.

Admission requires:

- Exact config, layer schedule, weight names/shapes, quantization modes and
  per-module exceptions; reject unsupported or malformed inputs explicitly.
- Forward shape, supported dtype and finite logits; correct routing and masks.
- Genuine cache prefill, continuation, advancing positions, reset and request
  isolation, with family-specific convolution/sliding/recurrent state.
- Basic generation, tokenizer/chat formatting and every configured EOS ID.
- Native Python, CLI, HTTP completion/chat and SSE smoke for advertised paths,
  including bounded queuing, cancellation, cleanup and reset/reload behavior.
- Clean wheel installation, dependency/import isolation, retained licenses and
  the ordinary existing regression suite, with native Metal checks where available.

Tiny synthetic configurations and fabricated safetensors test architecture
support without full model downloads. New architecture tests run by default
on a Metal host. Lack of full weights is never a reason to skip them. Other
hosts still run metadata, configuration and protocol tests.

| Status | Required evidence |
|---|---|
| `ARCHITECTURE_IMPLEMENTED` | Native config/model/cache/registry and passing tiny synthetic forward and cache tests |
| `MODEL_SMOKE_TESTED` | The exact downloaded checkpoint actually loads and generates through native code |
| `EXPERIMENTAL_SUPPORTED` | Required real-checkpoint serving/runtime smoke passes within disclosed bounds |
| `USER_VALIDATED` | The owner completes independent manual evaluation |

`CODE_IMPLEMENTED` and `SYNTHETIC_TESTED` are implementation milestones.
`NOT_DOWNLOADED_NOT_SMOKE_TESTED`, `NOT_DOWNLOADED` and `MANUAL_TEST_PENDING`
describe remaining checkpoint work. `INCOMPATIBLE_MLX_CORE` is a runtime
compatibility failure; MXFP4 must never be reinterpreted as affine Q4.

Full-vs-cached differential logit E, KL, NLL, extensive FP64 experiments and
exhaustive sequence-partition matrices remain **optional research diagnostics**.
They do not determine admission. Focused tolerance-based operation checks
remain useful when they test actual routing, masking, RoPE or quantization.
Known severe functional failures still block their affected advertised paths.

Gemma3 follows these ordinary functional rules. Its historical cached numerical
differences after sliding boundaries remain a known limitation, without a
numerical admission veto. Qwen3.5's current stored-source short generation and
continuation/reset recheck passed, but converted and previously broken paths
remain experimental and unqualified. A short source smoke does not promote them.

The packaged support manifest retains the previous ten exact checkpoint rows
and adds the two new large targets as architecture-only. Historical bounded
Python/CLI/HTTP/SSE evidence supports the nine existing experimental entries;
neither new target has real-checkpoint smoke evidence. Administrative publication
approval is separate from functional model admission and remains pending.

Configured position limits do not establish feasible context or hardware
performance. Start large-model manual smoke at 128–1024 total tokens and record
actual prompt/output counts, TTFT, throughput, MLX memory and state memory.

References: [MLX-LM](https://github.com/ml-explore/mlx-lm),
[current manifest](../src/mlx_one/data/compatibility/alpha_release.json),
[manual inference](native-moe/manual-inference.md),
[historical proof](qualification/alpha_release/model-support-matrix.md).
