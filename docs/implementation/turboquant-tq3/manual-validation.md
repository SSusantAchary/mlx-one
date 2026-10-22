# Owner validation: GPT-OSS-20B TQ3

The full checkpoint was **not downloaded or run** during implementation. Commands below use real
entrypoints/flags verified with a tiny fabricated TQ3 GPT checkpoint. The wheel, real CLI server,
Python/CLI/HTTP/chat/SSE, disconnect recovery and cancellation were tested. Large-model output,
practical context and performance remain **PENDING_USER_DOWNLOAD**.

Run from the MLX-One repository root on the M4/32GB host. Close other model processes before
changing backends. Keep the checkpoint's execution dtypes and FP16 KV; the `reference_dense` Engine
policy still executes native packed TQ3 projections. Do not select affine optimized conversion flags.

## Build and install a fresh wheel

The repository-root build and wheel/source audit were executed successfully. Build output is also
available now at `/tmp/mlx-one-tq3/root-dist/`. To reproduce:

```sh
.venv/bin/python -m build --no-isolation --outdir /tmp/mlx-one-tq3-build .
python3.12 -m venv .venv-tq3
.venv-tq3/bin/python -m pip install '/tmp/mlx-one-tq3-build/mlx_one-0.1.0a1-py3-none-any.whl[server]'
.venv-tq3/bin/python -m pip check
```

Native dependencies pin MLX/MLX-Metal 0.32.2. The fresh candidate environment needs neither
`mlx-lm`, `transformers` nor `turboquant-mlx-full`. The benchmark reference belongs in another venv.
Development tests/build tools remain in the existing `.venv`.

## Download — owner runs this after implementation

This downloads the actual large model. This command was **not executed** during implementation.

```sh
export TQ3_REV=20ba1835e625264e07366d215933fe31a9940a6a
export TQ3_DIR="$PWD/checkpoints/gpt-oss-20b-tq3/$TQ3_REV"
mkdir -p "$TQ3_DIR"
.venv-tq3/bin/hf download manjunathshiva/gpt-oss-20b-tq3 \
  --revision "$TQ3_REV" --local-dir "$TQ3_DIR"
```

Retain both shards, config, generation config, tokenizer JSON/config, chat template and shard index.
Do not rename packed tensors or convert this checkpoint with an affine quantizer.

## Validate metadata without loading weights

This validator reads only config/index and local safetensors headers. It can also validate the
already-fetched metadata fixtures without any downloaded model:

```sh
.venv-tq3/bin/python benchmarks/turboquant/validate_metadata.py \
  tests/fixtures/turboquant_metadata --output /tmp/tq3-metadata-fixture.json
.venv-tq3/bin/python benchmarks/turboquant/validate_metadata.py \
  "$TQ3_DIR" --output /tmp/tq3-local-metadata.json
```

Expected: `status=METADATA_VALIDATED`, `format=turboquant_tq3`, `tensor_count=966`,
`packed_modules=169`, `layers=24`, `experts=32`, `active_experts=4`,
`weight_payload_read=false`. Metadata validation does not grant model-smoke status. The native
loader also validates actual codebook/sign/scale contents and the strict parameter contract.

## Python, CLI, streaming and cancellation

Use an explicit final-channel prefix for a bounded first smoke. The checkpoint's chat template
normally leaves an open assistant header and permits analysis before final text.

```sh
export TQ3_PROMPT='<|start|>user<|message|>Say hello in one sentence.<|end|><|start|>assistant<|channel|>final<|message|>'
.venv-tq3/bin/mlx-one generate "$TQ3_DIR" "$TQ3_PROMPT" \
  --offline --max-tokens 32 --json-output
.venv-tq3/bin/mlx-one generate "$TQ3_DIR" "$TQ3_PROMPT" \
  --offline --max-tokens 32 --stream
```

```python
import os
import threading
from mlx_one import Engine, CacheConfig
from mlx_one.text import load_text_model, stream_chat, TextGenerationOptions

bundle = load_text_model(os.environ["TQ3_DIR"], offline=True)
for chunk in stream_chat(bundle, [{"role": "user", "content": "Say hello."}],
                         options=TextGenerationOptions(max_tokens=128, temperature=0)):
    print(chunk.text, end="", flush=True)
print()
with Engine(bundle, cache_config=CacheConfig(prefix_cache_enabled=False)) as engine:
    print(engine.generate(os.environ["TQ3_PROMPT"], max_tokens=32).to_json())
    cancel = threading.Event()
    stream = engine.stream(os.environ["TQ3_PROMPT"], max_tokens=128, cancel=cancel)
    next(stream)
    cancel.set()
    list(stream)
    stream.close()
    assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
    print(engine.generate(os.environ["TQ3_PROMPT"], max_tokens=32).text)
    print(engine.runtime_stats())
```

This uses one loaded model; close the Python process before starting the server or reference.
For the existing bounded functional smoke with structured metrics:

```sh
.venv-tq3/bin/python scripts/qualification/functional_checkpoint_smoke.py "$TQ3_DIR" \
  --prompt "$TQ3_PROMPT" --context 512 --max-tokens 32 --output /tmp/tq3-smoke-1.json
.venv-tq3/bin/python scripts/qualification/functional_checkpoint_smoke.py "$TQ3_DIR" \
  --chat --prompt 'Say hello in one sentence.' --context 512 --max-tokens 128 \
  --output /tmp/tq3-chat-smoke-1.json
```

Use new output filenames; that harness refuses to overwrite evidence. A tiny test's scripted
sampler is never used for owner validation. Actual visible output and EOS/counts must be recorded.

## HTTP, chat, SSE and disconnect recovery

Start one local server:

```sh
export MLX_ONE_API_KEY=local-test
.venv-tq3/bin/mlx-one serve "$TQ3_DIR" --offline --host 127.0.0.1 --port 8888 \
  --alias tq3 --api-key "$MLX_ONE_API_KEY" --context-length 512 \
  --inference-mode reference_dense --parallel 1 --queue-size 8 \
  --no-cache-prompt --no-warmup
```

In another terminal with the same key:

```sh
curl -fsS http://127.0.0.1:8888/v1/models \
  -H "Authorization: Bearer $MLX_ONE_API_KEY"
curl -fsS http://127.0.0.1:8888/v1/completions \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"tq3","prompt":"<|start|>user<|message|>Say hello.<|end|><|start|>assistant<|channel|>final<|message|>","max_tokens":32,"temperature":0}'
curl -fsS http://127.0.0.1:8888/v1/chat/completions \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"tq3","messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Say hello."}],"max_tokens":128,"temperature":0}'
curl -N -fsS http://127.0.0.1:8888/v1/chat/completions \
  -H "Authorization: Bearer $MLX_ONE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"tq3","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128,"temperature":0,"stream":true}'
curl -fsS http://127.0.0.1:8888/v1/runtime \
  -H "Authorization: Bearer $MLX_ONE_API_KEY"
```

SSE ends in `data: [DONE]`; terminal events carry `mlx` metrics. Nonstreaming responses contain
`usage` and `mlx`. Analysis is not emitted as final answer text. A short budget can legitimately
end before final text; compare actual generated count and finish reason before increasing it.

For cancellation, Ctrl-C a live `curl -N` or close its HTTP stream after the first event. Then send
a new completion and inspect `/v1/runtime` for recovery/no active handles. There is no invented
cancel/reset HTTP endpoint. Python event cancellation above is deterministic even if a tiny model
finishes before a TCP disconnect reaches the scheduler. Stop the server before benchmarking.

## Identical saved inputs and direct-model comparison

Create a sufficiently long numerical prompt workload using only the native tokenizer dependency:

```sh
.venv-tq3/bin/python - <<'PY'
import json, os
from pathlib import Path
from tokenizers import Tokenizer
root = Path(os.environ['TQ3_DIR'])
tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
ids = tokenizer.encode('Explain the role of expert routing in a language model. ' * 4000).ids
assert len(ids) >= 16384
Path('/tmp/tq3-input-ids.json').write_text(json.dumps(ids[:16384]))
Path('/tmp/tq3-api-prompt.txt').write_text(os.environ['TQ3_PROMPT'])
PY
```

These saved ID prefixes are repeatable numerical workloads. Use the chat calls for answer-quality
checks. Both direct backends consume exactly the saved IDs; no retokenization changes the inputs.

Prepare the isolated reference environment (this installs reference code, not model weights):

```sh
python3.12 -m venv .venv-tq3-reference
.venv-tq3-reference/bin/python -m pip install 'mlx==0.32.2' 'mlx-metal==0.32.2' \
  'mlx-lm==0.32.2' \
  'git+https://github.com/manjunathshiva/turboquant-mlx.git@54443c8b83424adb5dac7e6ffc7ac2d91fe88af1'
```

Run each backend sequentially in a fresh process with the same downloaded checkpoint. Begin with
128/512; profile layers separately from throughput samples (the driver does this automatically):

```sh
.venv-tq3/bin/python benchmarks/turboquant/full_model.py --backend native \
  --checkpoint "$TQ3_DIR" --checkpoint-revision "$TQ3_REV" \
  --token-ids /tmp/tq3-input-ids.json --input-lengths 128,512 --max-tokens 128 \
  --temperature 0 --top-p 1 --seed 42 --kv-dtype float16 --warmups 1 --repetitions 3 \
  --profile-layers --output /tmp/tq3-native-direct.json
.venv-tq3-reference/bin/python benchmarks/turboquant/full_model.py --backend reference \
  --checkpoint "$TQ3_DIR" --checkpoint-revision "$TQ3_REV" \
  --token-ids /tmp/tq3-input-ids.json --input-lengths 128,512 --max-tokens 128 \
  --temperature 0 --top-p 1 --seed 42 --kv-dtype float16 --warmups 1 --repetitions 3 \
  --profile-layers --output /tmp/tq3-reference-direct.json
.venv-tq3/bin/python benchmarks/turboquant/compare.py --native /tmp/tq3-native-direct.json \
  --reference /tmp/tq3-reference-direct.json --output /tmp/tq3-full-comparison.json
```

After short runs succeed, change **both** input matrices to `128,512,2048,8192,16384`. Use
`--skip-lengths 8192,16384` to omit stress cases. The driver checks working-set/available host memory,
conservative KV/scratch estimates and context. A skip is recorded, never a performance PASS.
Hashing the shards occurs before timed loading and verifies identical files across reports.
Cold TTFT includes first-inference compilation. Warmup costs are separate; steady-state samples
exclude warmups. Decode TPS uses actual decode steps after the first token, with EOS preserved.
The comparison rejects different hosts/MLX/weight hashes/prompt IDs/settings; unequal EOS counts
are marked incomparable instead of treating the requested budget as generated output.

JSON: `load_seconds`, `preflight_*`, `weight_sha256`, `results[].cold`, `warmup_seconds`,
`samples[].ttft_seconds/prefill_tokens_per_second/decode_tokens_per_second/generated_tokens/finish_reason`,
`cache_state_bytes`, `peak_mlx_active_bytes`, process RSS, dispatch counts, instrumented per-layer/MoE
samples, and `skipped`. Process wired memory is explicitly null when unavailable.
The actual measured comparison reports separate `decode_ratio`, `prefill_ratio` and `target_met`.
No such large-model result exists yet.

For Python Engine overhead, run the native driver separately with `--interface python` (omit
`--profile-layers`). It verifies decode/encode preserves IDs and otherwise directs you to the direct
interface. This is not mixed into direct-model results. For CLI and an already-running native HTTP
server, use separate interface reports:

```sh
.venv-tq3/bin/python benchmarks/turboquant/interfaces.py --interface cli \
  --cli "$PWD/.venv-tq3/bin/mlx-one" --checkpoint "$TQ3_DIR" \
  --prompt-file /tmp/tq3-api-prompt.txt --max-tokens 128 --repetitions 3 \
  --output /tmp/tq3-cli.json
.venv-tq3/bin/python benchmarks/turboquant/interfaces.py --interface http \
  --base-url http://127.0.0.1:8888 --model tq3 --prompt-file /tmp/tq3-api-prompt.txt \
  --max-tokens 128 --repetitions 3 --output /tmp/tq3-http.json
```

CLI wall time includes a fresh process and model load; HTTP wall time includes transport to a
resident server. Responses retain actual counts/finish reasons and native metrics. These costs
must not be presented as raw kernel throughput. Operator-only commands are in
[operator-benchmarks.md](operator-benchmarks.md).

## Troubleshooting and status promotion

* Malformed storage/codebook/sign errors: verify the pinned original files and index. Re-download
  incomplete shards; never reinterpret TQ3 as affine Q3 or apply a second weight rotation.
* No Metal device: run in a normal local Apple Silicon session with MLX/MLX-Metal 0.32.2.
* Context exceeds model maximum: reduce context; configured 131072 does not establish 32GB capacity.
* Memory preflight skip: start at 128/512, close other models, preserve FP16 KV, skip larger cases.
* Empty answer with generated tokens: inspect Harmony final-channel and EOS/budget metrics.
* HTTP 401: use the same Bearer key; 429 indicates the existing bounded queue/admission policy.
* Reference/native output counts differ: keep the comparison marked incomparable and inspect EOS,
  sampling and the pinned source/config differences; do not force a throughput PASS.
* Tiny/synthetic results establish backend correctness only. Promote to MODEL_SMOKE_TESTED and then
  EXPERIMENTAL_SUPPORTED only after the actual checkpoint passes functional load/generation/API
  validation. USER_VALIDATED requires the owner's evidence. The >=90% target is measured only by
  the actual same-host checkpoint run, not by operator tables.

Reference implementation difference: MLX-LM 0.32.2 rounds YaRN correction bounds, while native GPT
honors this checkpoint's `truncate=false`. The runner records this in comparison notes. Preserve
the difference when reporting results; inspect token/EOS divergence before making parity claims.
