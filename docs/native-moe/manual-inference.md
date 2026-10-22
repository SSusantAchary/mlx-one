# Local checkpoint manual inference

The commands below use existing repository entrypoints. CLI option names,
Python/HTTP paths and tiny-model behavior were checked during implementation.
The two large checkpoints are **NOT_DOWNLOADED / MANUAL_TEST_PENDING**;
commands using their weights are prepared for the owner and were not executed.
Load one model at a time. Start with a 512-token context; 131072 is only the
configured maximum, not demonstrated M4/32GB capacity.

## Environment and pinned downloads — owner runs later

From the repository root:

```sh
source .venv/bin/activate
python -m pip install -e '.[server]'
python -m pip check
export LFM_REV=667408d494bde191c2cb9dd4f4549827816b9275
export GPT_REV=f356f2747216d7e98fee755df25987459fc19089
export LFM_DIR="$PWD/checkpoints/lfm2.5-24b/$LFM_REV"
export GPT_DIR="$PWD/checkpoints/gpt-oss-20b/$GPT_REV"
mkdir -p "$LFM_DIR" "$GPT_DIR"
hf download LiquidAI/LFM2.5-24B-A2B-MLX-4bit \
  --revision "$LFM_REV" --local-dir "$LFM_DIR"
hf download mlx-community/gpt-oss-20b-MXFP4-Q4 \
  --revision "$GPT_REV" --local-dir "$GPT_DIR"
```

These last two commands download large weights. Neither ran during implementation.
Their commit-named directories pin the downloaded artifact. Once complete,
all inference commands below use local directories with networking disabled.
Retain config, generation config, tokenizer assets, template and weight index.

The tested runtime is `mlx==0.32.2` and `mlx-metal==0.32.2`. An older/incompatible
core produces `UNSUPPORTED_QUANTIZATION: INCOMPATIBLE_MLX_CORE` for MXFP4. Upgrade
both together in this environment and rerun the tiny tests:

```sh
python -m pip install --upgrade 'mlx==0.32.2' 'mlx-metal==0.32.2'
python -m pytest -q --no-cov \
  tests/models/test_large_moe_contracts.py tests/models/test_large_moe_native.py
```

## CLI and structured metrics

```sh
mlx-one generate "$LFM_DIR" 'The city was' --offline --max-tokens 32 --json-output
export GPT_PROMPT='<|start|>user<|message|>Say hello in one sentence.<|end|><|start|>assistant<|channel|>final<|message|>'
mlx-one generate "$GPT_DIR" "$GPT_PROMPT" --offline --max-tokens 32 --json-output
mlx-one generate "$GPT_DIR" "$GPT_PROMPT" --offline --max-tokens 32 --stream
```

The explicit GPT-OSS final header uses this artifact's actual older Harmony
token spellings. Its checkpoint template normally starts an open assistant
header and lets the model choose its channel. Nonfinal output is counted as
generated model tokens but never rendered as answer text. Chat can exhaust a
short budget before reaching a final answer; inspect counts before increasing
the budget up to 128 while keeping total context conservative.

Capture load, forward/cache continuation/reset, generation and request metrics:

```sh
python scripts/qualification/functional_checkpoint_smoke.py "$LFM_DIR" \
  --chat --prompt 'Say hello in one sentence.' --context 512 --max-tokens 64 \
  --output /tmp/lfm-native-smoke-1.json
python scripts/qualification/functional_checkpoint_smoke.py "$GPT_DIR" \
  --prompt "$GPT_PROMPT" --context 512 --max-tokens 64 \
  --output /tmp/gpt-native-smoke-1.json
python scripts/qualification/functional_checkpoint_smoke.py "$GPT_DIR" \
  --chat --prompt 'Say hello in one sentence.' --context 512 --max-tokens 128 \
  --output /tmp/gpt-native-chat-1.json
```

Use new output filenames on each run; the script never overwrites evidence.
Inspect `metrics.time_to_first_token_ms` (first model output token),
`server_text_ttft_ms` (first visible final text), `prompt_tokens_per_second`,
`decode_tokens_per_second`, `generated_tokens`, `memory_peak_bytes`,
`cache_logical_bytes`, `cache_allocated_bytes` and `cache_layers`. Measurements
are specific to the actual prompt and output budget. They are not benchmark
claims. A script PASS only establishes its recorded bounded operations.

## Python inference, chat, cancellation and reload

```python
import os
import gc
import threading
from mlx_one import CacheConfig, Engine
from mlx_one.text import TextGenerationOptions, load_text_model, stream_chat

path = os.environ["LFM_DIR"]  # switch to GPT_DIR in a separate run
bundle = load_text_model(path, offline=True)
chunks = stream_chat(bundle, [{"role": "user", "content": "Say hello."}],
                     options=TextGenerationOptions(max_tokens=128))
print("".join(chunk.text for chunk in chunks))

prompt = bundle.chat_template.render([{"role": "user", "content": "Say hello."}])
with Engine(bundle, inference_mode="reference_dense",
            cache_config=CacheConfig(prefix_cache_enabled=False)) as engine:
    print(engine.generate(prompt, max_tokens=128).text)
    cancel = threading.Event()
    stream = engine.stream(prompt, max_tokens=128, cancel=cancel)
    next(stream)
    cancel.set()
    list(stream)
    stream.close()
    assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
    print(engine.generate(prompt, max_tokens=32).text)  # fresh request/cache
    print(engine.runtime_stats())
del bundle, engine
gc.collect()

# The prior Engine is closed before loading the other model.
with Engine(os.environ["GPT_DIR"], offline=True,
            inference_mode="reference_dense",
            cache_config=CacheConfig(prefix_cache_enabled=False)) as engine:
    print(engine.generate(os.environ["GPT_PROMPT"], max_tokens=32).text)
```

## Authenticated, loopback-only HTTP

Choose **one** startup command, stopping the previous process before switching:

```sh
export MLX_ONE_API_KEY='replace-with-your-local-test-key'
mlx-one serve "$LFM_DIR" --offline --host 127.0.0.1 --port 8888 \
  --alias local-model --api-key "$MLX_ONE_API_KEY" --context-length 512 \
  --inference-mode reference_dense --parallel 1 --queue-size 8 \
  --no-cache-prompt --no-warmup
```

```sh
mlx-one serve "$GPT_DIR" --offline --host 127.0.0.1 --port 8888 \
  --alias local-model --api-key "$MLX_ONE_API_KEY" --context-length 512 \
  --inference-mode reference_dense --parallel 1 --queue-size 8 \
  --no-cache-prompt --no-warmup
```

In another terminal with the same API key:

```sh
curl -fsS http://127.0.0.1:8888/v1/models \
  -H "Authorization: Bearer $MLX_ONE_API_KEY"
curl -fsS http://127.0.0.1:8888/v1/completions \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"local-model","prompt":"The city was","max_tokens":32,"temperature":0}'
curl -fsS http://127.0.0.1:8888/v1/chat/completions \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
curl -N -fsS http://127.0.0.1:8888/v1/chat/completions \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0,"stream":true}'
curl -fsS http://127.0.0.1:8888/v1/runtime \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" > /tmp/mlx-one-runtime-1.json
```

For GPT-OSS raw completions use a Harmony prompt, e.g.:

```sh
python - <<'PY' > /tmp/gpt-completion-request.json
import json, os
print(json.dumps({"model":"local-model", "prompt":os.environ["GPT_PROMPT"],
                  "max_tokens":32,"temperature":0,"stream":True}))
PY
curl -N -fsS http://127.0.0.1:8888/v1/completions \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @/tmp/gpt-completion-request.json
```

Interrupt a streaming curl with Ctrl-C to test client disconnect; issue a new
request and verify runtime cache handles/reservations return to zero. Stop the
server with Ctrl-C, then launch the other local model. There is no fabricated
HTTP model-switch/reset endpoint. The server preserves its single-active
generation, bounded FIFO and existing cancellation/shutdown lifecycle.

## Licenses and limits

Adapted MLX-LM mathematics retain Apple MIT attribution and the packaged
`models/UPSTREAM_LICENSE`. Checkpoint metadata/templates retain source provenance
and license copies in test fixtures. GPT-OSS's model card declares Apache-2.0.
LiquidAI's `lfm1.0` license includes a commercial-use revenue threshold; consult
the exact checkpoint license for your use. The package includes no model weights.

- [LiquidAI artifact and license](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit/tree/667408d494bde191c2cb9dd4f4549827816b9275)
- [GPT-OSS artifact](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q4/tree/f356f2747216d7e98fee755df25987459fc19089)
- [Functional admission policy](../model-admission.md)
