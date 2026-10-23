# Native GPT-OSS / Kilo compatibility report

Tested on 2026-10-23 on Apple M4, 32 GB unified memory, Python 3.12, using the existing local `gpt-oss-20b-MXFP4-Q4` checkpoint. No checkpoint was downloaded. No inference-framework dependency was added. Changes are local and uncommitted.

## Baseline and source map

Branch: `chore/alpha-repository-cleanup`. Existing changes to `benchmarks/model_performance/report.py`, `tests/unit/test_model_performance_reporting.py`, and previously untracked optimization/qualification files were preserved.

Before edits, 37 relevant tests passed. Plain HTTP and SSE chat passed with the deterministic existing test server. A synthetic, sanitized 29-tool Kilo-shaped request reproduced HTTP 422 with `extra_forbidden` on `tools`, `tool_choice`, and `stream_options`. It also demonstrated that the old validation response echoed tool definitions unnecessarily.

The failure occurred before inference: `server/schemas.py:ChatCompletionRequest` forbade extra fields without modeling tools. Fixing validation alone would still lose tool actions: `text/harmony.py:HarmonyFinalParser` intentionally emitted only final text, the generation layer discarded terminal token identity, and HTTP/SSE serializers supported text only. Incoming tool messages were also unsupported.

| Responsibility | Existing source |
|---|---|
| Typed HTTP inputs | `src/mlx_one/server/schemas.py` |
| Auth, model validation, chat route, SSE | `src/mlx_one/server/app.py` |
| Native request adaptation, lease and cache lifecycle | `src/mlx_one/server/generation_engine.py` |
| Queueing, deadlines, cancellation, device ownership | `src/mlx_one/server/scheduler.py` (unchanged) |
| CLI serving options and native loading | `src/mlx_one/cli.py` (unchanged) |
| Served alias / model metadata | `src/mlx_one/server/model_manager.py` (unchanged) |
| Shared prompt preparation and context budget | `src/mlx_one/engine/prompts.py` |
| Checkpoint template dispatch | `src/mlx_one/text/chat.py` |
| Standalone checkpoint tokenizer | `src/mlx_one/text/tokenizers.py` (unchanged) |
| Native sampling and generated-token transport | `src/mlx_one/text/generation.py`, `text/schemas.py` |
| GPT-OSS model and mixed quantization | `src/mlx_one/models/language/gpt_oss/` (unchanged) |

## Exact files changed for this task

- `src/mlx_one/engine/prompts.py`
- `src/mlx_one/server/app.py`
- `src/mlx_one/server/generation_engine.py`
- `src/mlx_one/server/schemas.py`
- `src/mlx_one/text/chat.py`
- `src/mlx_one/text/generation.py`
- `src/mlx_one/text/schemas.py`
- `src/mlx_one/text/harmony_tools.py` (new)
- `tests/fixtures/kilo/request.json` (new, synthetic sanitized 29-tool fixture)
- `tests/test_gpt_oss_tools.py` (new)
- `tests/models/test_gpt_oss_tools_native.py` (new)
- `scripts/check_gpt_oss_tools.py` (new)
- `scripts/check_kilo_gpt_oss.py` (new)
- `docs/GPT_OSS_KILO_COMPATIBILITY.md` (this report)
- `docs/validation/gpt-oss-kilo-2026-10-23.json` (sanitized results)

Local integration configuration was also updated in `/Users/susant-achary/.config/kilo/kilo.jsonc`: only the existing `mlx_one` provider's `chunkTimeout` and this model's `temperature` / `reasoningEffort` options changed. All other settings and comments were preserved and the parsed result was checked against those three intended changes. Backup: `kilo.jsonc.mlx-one-backup-20261023-040516` beside the original file.

## Implementation

Request models explicitly support function definitions, JSON schemas, assistant tool calls, tool results, developer messages, tool selection, parallel-call control, and streaming usage. Nested schemas use Pydantic JSON values and retain arbitrary vocabulary, with bounded descriptions, catalog size, and schema nesting. Envelope models retain `extra="forbid"`. Function selection, duplicate names/IDs, orphan results, and missing tool results receive bounded validation errors. Text-only content-part arrays normalize to text instead of being misclassified as vision.

The native Harmony tool renderer places leading developer instructions before a single `functions` namespace in the same developer message, matching the checkpoint layout. It emits TypeScript-like argument declarations, required/optional fields, and only the additional schema constraints that signatures cannot express. It resolves control spellings against the checkpoint's single-token encoding. It retains the checkpoint's `commentary json` call headers and JSON encoding for text tool results. It reconstructs each assistant invocation and matches subsequent tool results by call ID and function name, preserving result order without double-encoding JSON arguments. Ordinary requests without tool settings retain their checkpoint template and existing text decoder.

The request-local `HarmonyToolParser` consumes actual token IDs, bypassing the legacy decoded-string filter for tool requests. It distinguishes analysis, commentary, final text, message ends, final completion, and tool handoff. Analysis is counted and withheld. Completed function arguments must be a JSON object with no duplicate keys or nonfinite numbers. Unknown function recipients, malformed output, and calls when disabled produce controlled generation failures, with fixed, bounded diagnostics and existing cleanup. Truncation returns `length` without publishing pending actions.

Actions are buffered until a validated handoff, then emitted through `GenerationEvent.tool_calls` as indexed metadata and ordered argument fragments. The nonstream serializer reconstructs the same calls; SSE emits `delta.tool_calls`, stable IDs, `finish_reason="tool_calls"`, optional final `choices: []` usage, and `[DONE]`. Empty text deltas keep the HTTP transport active during long prefill or hidden analysis without publishing reasoning or incomplete calls. Kilo's semantic chunk timer requires a separate timeout adjustment, described below. Tool execution remains entirely in Kilo.

The 29-tool hardware test exposed an existing native scheduling issue: ordinary, noncooperative execution could prefill the entire prompt despite a configured chunk size. A 6,982-token prompt forecast approximately 18.4 GB of temporary buffers and failed host admission before decoding. GPT-OSS tool requests now opt into bounded prefill chunks and progress events. Chunk-size probes use the existing allocation forecasts and reservation guards; critical pressure and every real forward remain guarded. No kernels, precision, scheduler, or KV ownership architecture changed. Context shifting removes whole user/tool turns instead of orphaning tool results. Visible text stops are applied after channel parsing and cannot truncate tool JSON.

## Automated results

- Baseline: **37 passed**.
- Final relevant HTTP/schema/parser/context suite, including the UI budget follow-up: **142 passed, 1 skipped**. The skip is the explicitly opt-in native completion module.
- Final native Metal suite: **31 passed**, including existing GPT-OSS/LFM MoE, GPT-2, native HTTP, and three new tool lifecycle tests.
- New deterministic protocol cases: **75 passed** (included in the 142 above).
- Ruff on changed Python files and `git diff --check`: passed.
- Only the existing FastAPI/Starlette deprecation warnings were observed.

The native fixtures execute actual tiny models with scripted token selection, exercising real forward passes, device ownership, repeated tool continuations, cancellation during partial output, malformed-output recovery, and zero remaining active KV handles. They need no full checkpoint. They are bounded regression tests, not the retired exhaustive numerical qualification suite.

### Built-in UI automatic reasoning budget follow-up

The user subsequently reported a chat HTTP 400 from the built-in UI. A request using the UI defaults reproduced `GPT-OSS reasoning switches and token budgets are not supported`: the UI explicitly sends `reasoning: "auto"` and `reasoning_budget: -1`. The GPT-OSS capability check incorrectly rejected the automatic `-1` sentinel whenever it was explicitly present. The route now resolves the effective request/server budget once and accepts `-1`, consistently with the generation settings. Explicit budgets and reasoning switches remain rejected.

Seventeen added HTTP cases cover plain chat and tools, streaming and nonstreaming, omitted and explicit automatic budgets, request precedence over server defaults, and continued rejection of unsupported controls. All passed as part of the 142-test suite. The reasoning subset also passed in the user's active Python 3.13 `(mlx)` environment (**24 passed** including existing reasoning cases). That environment imports this checkout directly; a running server must be restarted to load the correction. The full-checkpoint measurements below precede this validation-only correction and were not rerun.

## Real checkpoint results

`scripts/check_gpt_oss_tools.py` passed against the native server on port 8084:

| Check | Result |
|---|---|
| Authenticated discovery / served alias | PASS |
| Plain nonstream chat | PASS |
| Plain SSE chat | PASS |
| Auto function selection, structured nonstream call | PASS |
| Streamed function arguments reconstructed as JSON | PASS |
| Tool finish reason, terminal usage, `[DONE]` | PASS |
| Assistant call / matching tool result / meaningful continuation | PASS |
| 29-tool nested-schema catalog | PASS |
| Warmup plus five measured tool repetitions | PASS |
| Missing/wrong API key, invalid model, context overflow | PASS |
| Disconnect during native prefill / KV release / healthy follow-up | PASS |

Machine-readable local evidence: `runs/gpt-oss-kilo/http-hardware.json`, `startup.json`, `rss.json`. These ignored run artifacts contain no API credentials or private tool catalogs.

## Measured performance

The five small-tool repetitions ran sequentially without concurrent Kilo traffic. Figures apply to this checkpoint, machine, execution policy, and prompts only.

| Measurement | Observed |
|---|---:|
| CLI model load plus one-token warmup | 10.204 s |
| Plain SSE first visible text | 4.103 s |
| Small-tool first actionable delta, median | 7.127 s |
| Small-tool complete HTTP response, range | 7.100–7.141 s |
| Native model generation, median | 7.108 s |
| Native first sampled token, median | 2.355 s |
| Native prefill throughput, median | 75.95 tokens/s |
| Native decode throughput, median | 18.96 tokens/s |
| Small-tool queue wait, median | 1.05 ms |
| Small-tool peak MLX memory | 11,389,946,378 bytes |
| HTTP tool request plus continuation | 14.331 s |
| Synthetic 29-tool catalog prompt | 4,865 tokens |
| Synthetic 29-tool response including prefill | 89.873 s |
| Peak sampled process RSS during HTTP/cancellation checks | 5,183,700,992 bytes |
| MLX lifetime peak including the real Kilo task and direct follow-up | 12,184,864,734 bytes |
| Disconnect to zero active request KV handles | 7.068 s |

Startup includes filesystem cache effects and warmup; this is not a measured cold-disk load time. First actionable deltas intentionally wait for JSON validation. The HTTP round trip uses a synthetic client tool result and excludes real file/tool execution. Native generation and HTTP timings are recorded separately; their difference includes rendering, dispatch, and HTTP serialization. Process RSS is sampled separately from MLX memory and is not a lifetime peak. Kilo's actual execution and total elapsed time are recorded by its separate harness.

## Kilo coding-agent acceptance

The full coding-agent task **PASSED** with Kilo CLI 7.8.15. The CLI is installed only under `/private/tmp/mlx-one-kilo-cli`. The harness isolates XDG state, supplies an explicit config path and working directory, disables sharing, and denies external-directory tool access. It requests the factorial implementation, unittest creation/execution, repair if needed, and summary. Success requires at least two completed coding tools, created source/tests, Kilo running unittest, independent test success, and no client protocol errors.

The supplied `chunkTimeout: 120000` failed with this full catalog: Kilo resets that clock only for usable model output and retries while native prefill is still running. Empty SSE heartbeats do not reset its semantic timer. The isolated harness therefore uses **`chunkTimeout: 600000`** and deterministic model options **`temperature: 0`**, retaining the 1,200,000 ms overall timeout and the full catalog. After validation, these settings and `reasoningEffort: "medium"` were applied separately to the user's existing MLX-One provider, with the backup noted above.

Kilo also sends `reasoning_effort` on auxiliary title requests. Low, medium and high are explicitly modeled and applied to Harmony; other effort values return a capability error instead of being silently remapped.

The recorded run took **409.920 seconds**. It made **seven structured tool handoffs**, completed five tools, received two permission-denied tool results, corrected its requests, and finished with `stop`, with **zero client protocol errors**. Kilo created `hello.py` and `test_factorial.py`, ran its unittest successfully, and summarized the changes. An independent unittest rerun passed (one test method), and an additional check matched Python's `math.factorial` for inputs 0 through 10. No generated source or tests were edited by this assistant.

The first full-catalog response took 331.585 seconds with 13,937 prompt tokens. Subsequent tool responses took 8.278–17.597 seconds with prefix reuse. The eight HTTP responses totaled 406.878 seconds, while recorded Kilo tool executions totaled 0.200 seconds; the remaining wall time includes client startup and orchestration. These measurements include the temporary relay and are separate from the sequential small-tool measurements above.

Evidence: `runs/gpt-oss-kilo/kilo/result.json`, `summary.json`, `events.jsonl`, `protocol-summary.json`. The generated test project is `/private/tmp/mlx-kilo-34bchqwb/workspace`. A subsequent real Kilo session ran unittest **directly against port 8084**, without the relay, and passed with no client errors (`direct-result.json`, 326.643 seconds).

A temporary localhost relay records only request field names, role sequence, tool count, finish reason, usage, and function names to diagnose client behavior. It never executes tools or logs authorization, prompts, reasoning, or argument bodies. This CLI version sends 23 tools; the separate deterministic/hardware fixture covers the requested 29-tool shape.

## Limitations and compatibility

| Capability | Status |
|---|---|
| Native GPT-OSS text, auto tools, tool results | Implemented; hardware tested |
| SSE tool calls / usage / model alias / bearer auth | Implemented; tested |
| `tool_choice="none"` | Enforced; no actionable invocation may escape |
| `parallel_tool_calls=false` | Enforced, at most one call |
| Multiple calls in one response | Parser/HTTP synthetic support; native parallel batching unqualified |
| `required` / forced named function | Explicit HTTP 400 capability error |
| `strict=true` schema-constrained sampling | Explicit HTTP 400 capability error |
| GPT-OSS analysis | Native behavior preserved; not exposed by default |
| GPT-OSS `reasoning_effort` low / medium / high | Implemented in the native Harmony prompt |
| Other reasoning effort values, switches / token budgets | Explicitly unsupported |
| Tool payload sizes | Existing 1 MiB text body limit; at most 128 tools |
| 131,072 configured context | Configuration retained; 128K inference reliability unvalidated |
| Kilo full coding task | PASS with temperature 0, medium effort, 600-second chunk timeout |
| llama.cpp performance comparison | Not run |

## Reproduction

From the repository root, with the existing environment:

```bash
.venv/bin/python -m pytest --no-cov -q \
  tests/test_gpt_oss_tools.py tests/test_server.py tests/test_server_media.py \
  tests/test_server_controls.py tests/test_server_scheduler.py tests/api \
  tests/alpha_0_1/test_server_contract.py tests/test_text_runtime.py

MLX_ONE_RUN_MLX_TESTS=1 .venv/bin/python -m pytest --no-cov -q \
  tests/models/test_gpt_oss_tools_native.py tests/models/test_large_moe_native.py \
  tests/api/test_completions_native.py tests/test_native_gpt2_mlx.py

.venv/bin/mlx-one serve ~/mlx-one-models/gpt-oss-20b-MXFP4-Q4 \
  --offline --host 127.0.0.1 --port 8084 \
  --alias gpt-oss-20b-mxfp4-128k --api-key local-llama \
  --context-length 131072 --timeout 1200

curl -sS http://127.0.0.1:8084/v1/models \
  -H 'Authorization: Bearer local-llama'

MLX_ONE_API_KEY=local-llama .venv/bin/python scripts/check_gpt_oss_tools.py \
  --report runs/gpt-oss-kilo/http-hardware.json

npm install --prefix /private/tmp/mlx-one-kilo-cli \
  --cache /private/tmp/mlx-one-kilo-npm-cache --no-audit --no-fund @kilocode/cli@7.8.15

MLX_ONE_API_KEY=local-llama .venv/bin/python scripts/check_kilo_gpt_oss.py \
  --kilo /private/tmp/mlx-one-kilo-cli/node_modules/.bin/kilo \
  --chunk-timeout 600000 --report-dir runs/gpt-oss-kilo/kilo
```

The Kilo harness leaves its disposable workspace and transcript/report paths for inspection. The harness itself does not modify the user's global Kilo configuration. For an existing Kilo installation, pass `--kilo kilo`. The normal interactive provider configuration in the task remains applicable with provider `chunkTimeout` set to `600000` and model options set to `{"temperature": 0, "reasoningEffort": "medium"}`; `reasoning: true` in model metadata does not enable raw reasoning response metadata.

Protocol references consulted: [OpenAI Harmony format](https://developers.openai.com/cookbook/articles/openai-harmony), [Harmony renderer](https://github.com/openai/harmony), [function calling](https://developers.openai.com/api/docs/guides/function-calling), [Kilo custom models](https://github.com/Kilo-Org/kilocode/blob/main/packages/kilo-docs/pages/code-with-ai/agents/custom-models.md), [MLX-LM GPT-OSS](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gpt_oss.py), [Transformers GPT-OSS](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_oss), and [llama.cpp chat interoperability](https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp). Implementation is native MLX-One code; none of those serving frameworks is required at runtime.
