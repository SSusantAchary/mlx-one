# Native Web UI and server

mlx-one includes a local OpenAI-compatible server and a static browser UI. The request path is
entirely native:

```text
browser → mlx-one server → mlx-one model/generation APIs → MLX
```

The server does not import or invoke `mlx-lm`, llama.cpp, Ollama, or another inference server.

## Run

```bash
mlx-one serve mlx-community/Qwen3.5-0.8B-4bit
```

The default API base is `http://127.0.0.1:8888/v1` and the UI is at
`http://127.0.0.1:8888`. Use `--host` and `--port` to change the listener. The command also
accepts the same `--revision`, `--offline`, and `--cache-dir` controls as other model commands.
For a checkpoint without tokenizer assets, pass `--tokenizer REPOSITORY_OR_DIRECTORY`.

Common native serving controls:

```bash
mlx-one serve MODEL \
  --alias local-model \
  --api-key local-secret \
  -c 32768 \
  -np 2 \
  --queue-size 8 \
  --prefill-chunk-size auto \
  --max-batch-tokens auto \
  --prefix-cache-mib 512 \
  --timeout 600 \
  --warmup \
  --cache-prompt \
  --cache-reuse 256 \
  --context-shift \
  --kv-cache-bits 4
```

Enable private, local microphone and audio-file transcription by explicitly loading a native
Whisper sidecar:

```bash
mlx-one serve CHAT_MODEL \
  --transcription-model WHISPER_MODEL \
  --transcription-revision REVISION
```

The sidecar is not exposed as a second chat model and is never downloaded implicitly. The Web UI
enables its microphone and audio-upload controls only after `/v1/runtime` reports transcription
support. Recordings are limited to five minutes and uploads to 25 MiB. The returned transcript is
inserted into the composer for editing and is never sent automatically. FFmpeg must be installed
when the sidecar is enabled.

`MLX_ONE_API_KEY` is an alternative to `--api-key`, and the option can be repeated. API keys
protect `/v1/*`; health and the UI shell remain public. Enter the key in the UI's settings and
press Connect. The browser keeps it in memory only.

The server defaults to one cooperative generation slot, eight queued requests, a 600-second
whole-request deadline, startup warmup, prompt caching, native-precision KV state, and no
context shifting. `-ctk` and `-ctv` independently accept `f16`, `bf16`, `q4_0`, or `q8_0`.
Quantization applies only to attention KV tensors; convolution and recurrent state keep their
required native precision.

V1 loads one model per process. It supports GPT-2, Qwen2, Qwen2-MoE, Qwen3, LFM2, LFM2-MoE,
OpenELM, and Qwen3.5 in text-only mode. A registered architecture is not automatically a claim
that every checkpoint has completed qualification.

## API

- `GET /health` reports server readiness.
- `GET /v1/models` returns the loaded model using the OpenAI list shape.
- `GET /v1/runtime` returns measured model, context, timing, MLX memory, and transcription
  capability information.
- `POST /v1/chat/completions` supports text messages, streaming, temperature, top-p, top-k,
  maximum tokens, seed, and stop strings.
- `POST /v1/embeddings` and `POST /v1/rerank` use OpenAI-compatible and documented
  mlx-one response shapes when the loaded runner advertises those capabilities.
- `POST /v1/audio/transcriptions` accepts authenticated multipart audio when the private native
  Whisper sidecar is loaded.

Streaming responses are SSE records followed by `data: [DONE]`. Stop in the UI aborts the HTTP
request, which cooperatively cancels native generation. Requests execute through one MLX worker
and a bounded FIFO queue. `-np/--parallel` interleaves token steps fairly on that worker; it does
not create additional MLX inference threads.

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-secret' \
  -d '{"model":"mlx-community/Qwen3.5-0.8B-4bit","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

Reasoning-capable templates can be controlled with `--reasoning auto|on|off`,
`--reasoning-format`, `--reasoning-budget`, and `--reasoning-preserve`. The default `deepseek`
format streams reasoning through `delta.reasoning_content`. Request bodies may override
`reasoning` and `reasoning_budget`.

`deepseek-legacy` keeps `<think>` tags in normal content while also filling
`reasoning_content`. `--reasoning-preserve` passes preservation controls to compatible chat
templates and accepts prior assistant `reasoning_content` on subsequent requests; the browser
keeps that history in memory only.

Qwen3.5 checkpoints containing the complete native `mtp.*` tensor group can use
`--spec-type draft-mtp --spec-draft-n-max 3`. Greedy requests use MTP drafting and verification;
sampled requests safely fall back to ordinary decoding. Missing or partial MTP weights are an
explicit startup error when speculation is requested.

Tools, agents, and remote model routing remain outside the compatibility target. Multimodal chat
content is accepted only for a capability-qualified vision runner. Images are limited to four per
request and 20 MiB decoded each, and must be JPEG, PNG, or WebP data URLs or public HTTPS URLs.
Remote retrieval rejects credentials, private/link-local/reserved destinations, unsafe redirects,
oversized responses, MIME mismatches, and invalid image bytes. Server filesystem paths are never
accepted from HTTP clients.

## Development

Backend:

```bash
python -m pip install -e '.[dev]'
mlx-one serve MODEL
```

Frontend:

```bash
cd ui
npm install
npm run dev
```

Vite runs on port 5173 and proxies `/health` and `/v1` to `http://127.0.0.1:8888`. Override the
target with `MLX_ONE_SERVER_ORIGIN`.

Run checks with:

```bash
pytest
ruff check .
cd ui
npm run check
npm test
npm run build
npm run test:e2e
```

The E2E command starts a fake native engine through the real FastAPI server and opens the
packaged UI in Playwright Chromium, so it validates streaming and Stop without model downloads.

The SvelteKit static adapter writes production files to `src/mlx_one/ui/dist`. These files are
included as Python package data, so wheel and sdist users do not need Node.js. Maintainers must
rebuild and commit the generated directory whenever frontend sources change.

Build distributions with `python -m build`, then verify that both archives contain
`mlx_one/ui/dist/index.html` and its `_app` assets. An installed wheel can run `mlx-one serve`
without Node.js or a frontend checkout.

## Dependencies and security

`tokenizers` and Jinja2 are runtime dependencies. Install FastAPI and Uvicorn only for serving:

```bash
python -m pip install 'mlx-one[server]'
```

All model execution remains inside mlx-one's native Apple MLX implementations.
`mlx-lm` is not a serving dependency; the optional `legacy-mlx-lm` extra exists
only for compatibility workflows outside this server path.

The server binds only to loopback by default, applies a 1 MiB text JSON limit, a 32 MiB
multimodal-chat limit, and a 25 MiB audio-file limit, accepts development CORS only from the local
Vite origins, and never accepts model or filesystem paths through its HTTP API. Binding to a public
interface is an explicit operator decision; V1 is not a hardened multi-user gateway.

Four-bit loading accepts only standard MLX metadata with `bits: 4`, a positive group size, and
the `affine`, `mxfp4`, or `nvfp4` mode. Modules are quantized only when matching scale tensors
exist; mixed checkpoints are supported, while non-MLX formats and incomplete packed tensor
sets fail validation before inference.

## Terminal runtime logs

`mlx-one serve` prints elapsed timestamps and lifecycle/timing logs at `info`
level by default. The elapsed format is `minutes.seconds.milliseconds.microseconds`.
Severity and timestamps are colored on a terminal; redirected logs are plain
text. Set `NO_COLOR=1` to disable runtime-log color explicitly.

Startup logs report model loading, warmup, architecture, quantization, slots,
context limit, KV types, prefix-cache budget, execution selection and endpoints.
Uvicorn continues to report actual HTTP startup/shutdown and access status.
Text requests have slot/task identifiers, periodic decode progress, final prompt
evaluation and decode rates, total time, model TTFT, visible-text TTFT, cache
reuse, context used and observed active MLX memory. Warmup requests are bracketed
by explicit warmup-start and warmup-completed events. Task IDs identify engine
invocations, including warmup and retried attempts; they are not client request IDs.
Vision-language requests log available terminal metrics; text-token progress
requires the native text execution observer.

Use the normal serving command with either of these optional controls:

```bash
mlx-one serve MODEL --log-level info --log-progress-interval 3
mlx-one serve MODEL --log-progress-interval 0  # terminal summaries, no progress
mlx-one serve MODEL --log-level warning      # warnings/errors only
```

Decode progress consumes existing token timestamps and is throttled by time;
it does not synchronize Metal, hash sources, collect telemetry or print on every
token. Prompt-evaluation time is the sum of synchronized prefill forwards over
actually evaluated tokens. Decode rate uses first-to-last model token intervals;
`forward` is model-forward time only. Overall rate includes the whole engine
request. Queue waiting and network/client rendering are outside those engine
rates. Recent decode rate covers the last progress window. Model-token TTFT and
first-visible-text TTFT are separate because hidden reasoning or output handling
can delay visible text. No graph-reuse/reallocation counters are printed because
this runtime does not measure them. Missing timings or memory observations are
omitted rather than inferred.

Logs contain timing/count/configuration information, not prompts, generated
text, tool arguments, API keys or exception bodies. Failure logs include the
exception class and interrupted/cancelled requests have a terminal status.
The logger wraps generation with explicit iterator cleanup, preserving cache and
model-lease cleanup on client disconnect. Other task-specific servers retain
their existing access logs.

## Chat workspace and per-turn metrics

The bundled UI uses a dark workspace with desktop navigation, session conversation
history/search, closeable tabs, connected-model details, and settings/system-message
panels. The navigation becomes a slide-out menu on phones. A light theme remains
available through the theme button.

Conversations are kept in browser memory only. Reloading clears chats and API keys;
there is no new browser storage, cloud sync, or server-side history database.
Switching and closing conversations are disabled while generation or transcription
is active so streamed output cannot land in the wrong tab.

Each message offers copy, edit, branch, and delete-from-here actions. Assistant
responses can be regenerated. Editing a user message regenerates the continuation;
editing an assistant message saves the edited text. Both editing and regeneration
replace the subsequent continuation. Branching creates a separate conversation and
preserves the original. Deleting from a message removes it and subsequent messages.

Completed responses keep their own returned metrics and model metadata; later
requests do not overwrite earlier turn measurements. User rows show the associated
request's evaluated input count, synchronized prefill-forward duration and prefill
rate when available. These are request-input counts, not an estimate of the text
in the user's single bubble. Assistant rows show generated tokens, synchronized
decode-forward duration, first-to-last-token decode throughput and model TTFT.
Forward duration and decode rate use different time windows, as labeled. The footer
separates decode rate from whole-request throughput.

The context button beside the composer opens a usage panel. It shows the last
completed request's context usage and configured capacity, fresh/reused input,
output tokens, and accumulated completed-turn input/output counts. Its average
speed is total generated tokens divided by summed engine durations of turns with
timing data. Last-request context is not a claim that its KV cache remains resident:
retention and prefix reuse still depend on the server configuration. Missing
measurements show a dash. Models and quantization badges use backend metadata;
no model size or benchmark result is fabricated.

The Models panel lists models already exposed by the server. It does not download
weights or dynamically load additional checkpoints. MCP is explicitly marked
unavailable: this server has no MCP connection-management API. External clients
such as Kilo can provide their own tool integrations. Audio recording/upload,
local transcription, streamed Markdown sanitization and generation cancellation
continue to use their existing API paths.

The production assets are built from `ui/` into `src/mlx_one/ui/dist/` using
`npm ci && npm run build`. Restart with the local patched source and hard-refresh
the browser after updating. Validation uses a fake local backend, not a model
performance benchmark: 14 component/service tests, 9 browser scenarios, and Svelte
checking with zero errors/warnings cover streaming/stop, audio upload, scrolling,
per-turn metrics/context, editing/regeneration, branching/history/search, model
panels and mobile navigation.

Live combined verification on 14 November 2026 used the actual authenticated
GPT-OSS server through a browser: a complete 209-token response measured 32.16
model decode tok/s and 29.74 overall tok/s, with 528.67 ms model TTFT and
3687.68 ms visible-text TTFT. Terminal summaries matched API metrics, the usage
popover and model/cache details rendered, and no browser page errors occurred.
This is a bounded functional smoke, not a long-context or Kilo-agent qualification.
The initial-runtime/fast-response merge race found during testing is covered by
a browser regression test.

## Streaming progress and output limits

GPT-OSS chat uses the same token-aware Harmony parser with or without tools.
It counts analysis tokens while returning only visible answer content. During
streaming, `mlx_progress` frames report `generated_tokens`, `phase` (generating or
reasoning) and, for GPT-OSS, `reasoning_tokens`. Updates are throttled to at most
four per second and contain no reasoning text or tool arguments. The UI also
shows Processing prompt before the first model token. Other OpenAI-compatible
clients may ignore this optional SSE extension.

`max_tokens` is the total generated-token budget, including reasoning and protocol
tokens, not a guarantee of that many visible answer tokens. The UI now keeps a
visible Thinking/Generating status while content is hidden, displays a terminal
notice for `finish_reason=length`, and retains partial output on interruption.
A stream that ends without a terminal finish reason is reported as interrupted
rather than silently successful.

Retry with more tokens replaces the limited continuation, raises the configured
output budget to at least 2048 (or doubles it when larger), and caps it at the
remaining configured context. It does not append an empty assistant message or
claim to recover hidden analysis. Empty assistant outputs are excluded from later
request history. Model quality and the chance of exhausting a larger budget remain
separate from transport liveness; a larger cap is not a completion guarantee.

Validation: 110 backend tests, 9 UI unit/service tests and 8 browser scenarios
passed. A live authenticated browser test reproduced a 32-token reasoning-only
cutoff: Thinking progress was visible, 29 reasoning tokens were reported, and
retrying with a 2048-token budget completed with a normal stop (210 generated,
94 reasoning tokens). Both requests used the same prompt and existing model;
no reasoning text was exposed. This is bounded functional evidence, not a new
long-context benchmark. Open a fresh tab to load rebuilt assets while retaining
an older session's conversation in its original tab.

## GPT-OSS coding defaults

On first connection to a GPT-OSS model, the UI now recommends 8192 output tokens
and low reasoning effort. The recommendation is bounded by the configured context
with space reserved for input. An explicitly edited output budget is preserved,
even if it is 512; subsequent Connect clicks do not reset it. Other model defaults
are unchanged. The GPT-OSS Settings panel exposes low/medium/high reasoning effort
instead of the unrelated generic thinking-marker budget. Higher effort and explicit
small limits can still exhaust the budget before the final answer.

These defaults apply to fresh UI sessions. Existing loaded tabs keep their current
settings; open a new tab to load rebuilt assets while retaining an old conversation.
Check Max tokens and Reasoning effort before testing long coding prompts.

A live UI test used the exact prompt “code fastapi server for 1000 concurrent user
to accept request for embedding generation” with max_tokens=4096 and low effort.
Visible code started after 2.179 s as observed by the browser (server visible-text
TTFT 1.918 s). It ended normally with 2016 generated tokens including 34 reasoning
tokens, versus the previously observed reasoning-only 512-token cutoff. The whole
engine request took 125.379 s and decoded at 16.14 tok/s on this long run; this is
not a claim of sustained 30+ tok/s. No load test of the generated FastAPI program,
embedding model, or 1000 concurrent clients was performed. Generated code quality
and capacity claims are separate from this streaming/UI check.

Validation: 12 UI unit/service tests, 9 browser scenarios and Svelte checking with
zero errors/warnings passed. Coverage includes preset transmission and preservation
of manually selected limits. Raw live records are under ignored
`local/coding-defaults/`.

The current GPT-OSS recommendation has since increased to **8192 output tokens**
(with the same context bound and explicit-budget preservation). Earlier 4096-token
measurements above are historical. Fresh tabs receive the new recommendation;
existing tabs can change Max tokens to 8192 in Settings. This raises an allowance,
not a minimum response length or a guarantee of faster generation. Long-output
throughput remains under investigation; see the long-response results in
`docs/mlxboost/long-response-results.json`.
