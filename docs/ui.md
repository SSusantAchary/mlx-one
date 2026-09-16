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

The default API base is `http://127.0.0.1:8080/v1` and the UI is at
`http://127.0.0.1:8080`. Use `--host` and `--port` to change the listener. The command also
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
curl http://127.0.0.1:8080/v1/chat/completions \
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

Vite runs on port 5173 and proxies `/health` and `/v1` to `http://127.0.0.1:8080`. Override the
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
