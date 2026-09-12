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

V1 loads one model per process. It supports GPT-2, Qwen2, Qwen2-MoE, Qwen3, LFM2, LFM2-MoE,
OpenELM, and Qwen3.5 in text-only mode. A registered architecture is not automatically a claim
that every checkpoint has completed qualification.

## API

- `GET /health` reports server readiness.
- `GET /v1/models` returns the loaded model using the OpenAI list shape.
- `GET /v1/runtime` returns measured model, context, timing, and MLX memory information.
- `POST /v1/chat/completions` supports text messages, streaming, temperature, top-p, top-k,
  maximum tokens, seed, and stop strings.

Streaming responses are SSE records followed by `data: [DONE]`. Stop in the UI aborts the HTTP
request, which cooperatively cancels native generation. Requests execute through one MLX worker
and a bounded FIFO queue.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.5-0.8B-4bit","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

This is a deliberately limited compatibility target. Tools, multimodal message content, agents,
authentication, and remote model routing are rejected or unavailable in V1.

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
```

The SvelteKit static adapter writes production files to `src/mlx_one/ui/dist`. These files are
included as Python package data, so wheel and sdist users do not need Node.js. Maintainers must
rebuild and commit the generated directory whenever frontend sources change.

Build distributions with `python -m build`, then verify that both archives contain
`mlx_one/ui/dist/index.html` and its `_app` assets. An installed wheel can run `mlx-one serve`
without Node.js or a frontend checkout.

## Dependencies and security

FastAPI, Uvicorn, `tokenizers`, and Jinja2 are runtime dependencies. Legacy workflows that still
bridge to `mlx-lm` require:

```bash
python -m pip install 'mlx-one[legacy-mlx-lm]'
```

The server binds only to loopback by default, applies a 1 MiB request limit, accepts development
CORS only from the local Vite origins, and never accepts model or filesystem paths through its
HTTP API. Binding to a public interface is an explicit operator decision; V1 is not a hardened
multi-user gateway.

Four-bit loading accepts only standard MLX metadata with `bits: 4`, a positive group size, and
the `affine`, `mxfp4`, or `nvfp4` mode. Modules are quantized only when matching scale tensors
exist; mixed checkpoints are supported, while non-MLX formats and incomplete packed tensor
sets fail validation before inference.
