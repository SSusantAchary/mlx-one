# Native VLM checkpoint testing on Apple M4 32 GB

The follow-up hardware campaign uses pinned Hugging Face documentation images;
see [current real-image measurements and reproduction](benchmarks/vlm/README.md#real-checkpoint-hf-documentation-image-protocol-2026-10-24).
The original synthetic-image instructions below remain a separate protocol.

V01–V05 have now passed real **CHECKPOINT_SMOKE_PASSED** checks on the pinned
primary revisions, with actual HF-image benchmark evidence. Tiny synthetic source
tests remain a separate validation layer. The commands below reproduce the
manual checks after download; run one checkpoint at a time.
Follow the functional [model admission policy](model-admission.md); the retired
exhaustive numerical qualification suite is not required or reinstated.

## Install

From the repository root:

```bash
python -m pip install -e '.[vision,server]'
mkdir -p ~/mlx-one-models/vlm
```

Use Python 3.12 on Apple Silicon with the repository-pinned MLX/Metal 0.32.2.
Do not run the GPT-OSS/llama.cpp server or another GPU workload while benchmarking.
The following are **manual downloads**, not commands run during implementation.
Keep all configuration, tokenizer, processor and weight files together. `--revision`
pins an immutable commit; the benchmark also verifies the actual publisher weight
SHA-256 and original config/processor/template/tokenizer identities. A user-supplied
revision string alone is not sufficient performance provenance.

## Pinned primary downloads

### V01 — Qwen2-VL 2B (Q4)

```bash
hf download mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --revision 01af461cdb9574acc09084a0ef94e216e142b085 \
  --local-dir ~/mlx-one-models/vlm/qwen2-vl-2b
```

### V02 — Qwen2.5-VL 3B (Q4)

```bash
hf download mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --revision 46d4cf06a06ffc1a766c214174f9cbed2f45bcab \
  --local-dir ~/mlx-one-models/vlm/qwen2.5-vl-3b
```

### V03 — Qwen3.5 0.8B (Q4)

```bash
hf download mlx-community/Qwen3.5-0.8B-4bit \
  --revision da28692b5f139cb0ec58a356b437486b7dac7462 \
  --local-dir ~/mlx-one-models/vlm/qwen3.5-0.8b
```

### V04 — LFM2-VL 450M (Q8)

```bash
hf download mlx-community/LFM2-VL-450M-8bit \
  --revision 2422c0686f1f1c6267640bc1f069ff4e8b06b8d0 \
  --local-dir ~/mlx-one-models/vlm/lfm2-vl-450m
```

### V05 — LFM2.5-VL 3B (Q4)

```bash
hf download LiquidAI/LFM2.5-VL-3B-MLX-4bit \
  --revision 7fad79d26092c1145aa8dde873896a07d3e96a3b \
  --local-dir ~/mlx-one-models/vlm/lfm2.5-vl-3b
```

For alternatives, inspect their separate `source.json` metadata fixtures rather
than reusing the primary checkpoint's hashes. Prepared pins:

- `mlx-community/LFM2-VL-3B-4bit` at `aa86be09cd3bcdfd825252ce702d6afefc1b4201` (later manual testing).
- `mlx-community/Qwen3.5-0.8B-MLX-4bit` at `5d894f8cc4ef3e6c88537bf3746ed262f549da6a` (later manual testing).

## Native Python image inference

This example uses the versioned benchmark image, so it is executable without
supplying another image. Run from the repository root after the V01 download:

```python
from pathlib import Path
from mlx_one.text import TextGenerationOptions
from mlx_one.vision import VisionLanguageEngine

model = Path('~/mlx-one-models/vlm/qwen2-vl-2b').expanduser()
image = Path('benchmarks/vlm/images/VLM-S1-1.png')
messages = [{'role': 'user', 'content': [
    {'type': 'image', 'image': image},
    {'type': 'text', 'text': 'Describe the shapes and colors.'},
]}]
with VisionLanguageEngine(model, offline=True) as engine:
    result = engine.generate(messages=messages, options=TextGenerationOptions(max_tokens=64))
    print(result.text)
    print(result.token_ids, result.prompt_tokens, result.generation_tokens)
    print(result.finish_reason, result.image_metadata, result.timings)
    stream = engine.stream(messages=messages, options=TextGenerationOptions(max_tokens=64))
    try:
        for chunk in stream:
            print(chunk.text, end='', flush=True)
    finally:
        stream.close()
```

`engine.generate('Describe this image.', images=[image])` is the simple one-turn
form. Both forms apply the checkpoint chat template. The lower-level
`load_vlm_model`/`stream_vlm` functions remain available for explicitly rendered
raw prompts with one `<image>` sentinel per image; they do not own a service
thread. Use the managed engine for application concurrency and cleanup.

Local PNG/JPEG/WebP, PIL images and encoded image bytes are supported. Caller PIL
images remain owned by the caller. Content arrays preserve text/image order and
all conversation turns. Each request re-encodes its images once during prefill;
decode steps reuse the private complete model state. For multi-turn image chat,
send the image in the history again; no cross-request media cache is assumed.
Close a stream early or pass a `threading.Event` as `cancel=` to stop generation.

The following checks Python, streaming early close, two-turn image history and
10 repeated requests, reporting post-request active MLX memory:

```bash
python scripts/vlm_manual_smoke.py --mode python \
  --checkpoint ~/mlx-one-models/vlm/qwen2-vl-2b \
  --revision 01af461cdb9574acc09084a0ef94e216e142b085 \
  --repetitions 10
```

## Native CLI image inference

```bash
mlx-one vlm --model ~/mlx-one-models/vlm/qwen2-vl-2b \
  --offline --image benchmarks/vlm/images/VLM-S1-1.png \
  --prompt 'Describe the shapes and colors.' --max-tokens 64 --json-output

mlx-one vlm --model ~/mlx-one-models/vlm/qwen2-vl-2b \
  --offline --image benchmarks/vlm/images/VLM-S1-1.png \
  --image benchmarks/vlm/images/VLM-S3-2.png \
  --prompt 'Compare these images.' --max-tokens 64 --stream
```

`--image` is repeatable; Ctrl+C stops streaming and closes the service. The
existing `mlx-one generate` command remains a text-completion command.

## HTTP image chat and SSE

The existing positional `serve MODEL` convention, alias, authentication,
scheduler and native generation engine are preserved. This command's startup,
warmup and image dispatch were exercised for all five **tiny synthetic** variants;
real downloaded checkpoints still require your manual smoke:

```bash
mlx-one serve ~/mlx-one-models/vlm/qwen2-vl-2b \
  --offline \
  --host 127.0.0.1 \
  --port 8086 \
  --alias qwen2-vl-2b \
  --api-key local-llama
```

In a second terminal, generate an actual OpenAI-compatible image request:

```bash
python - <<'PYCODE'
import base64, json
from pathlib import Path
image = base64.b64encode(Path('benchmarks/vlm/images/VLM-S1-1.png').read_bytes()).decode()
payload = {'model': 'qwen2-vl-2b', 'max_tokens': 64, 'stream': False,
           'messages': [{'role': 'user', 'content': [
               {'type': 'text', 'text': 'Describe the shapes and colors.'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}}]}]}
Path('/tmp/mlx-one-vlm-request.json').write_text(json.dumps(payload))
PYCODE
curl --fail-with-body -sS http://127.0.0.1:8086/v1/chat/completions \
  -H 'Authorization: Bearer local-llama' \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/mlx-one-vlm-request.json

python - <<'PYCODE'
import json
from pathlib import Path
p = Path('/tmp/mlx-one-vlm-request.json')
v = json.loads(p.read_text())
v['stream'] = True
p.write_text(json.dumps(v))
PYCODE
curl --fail-with-body -N http://127.0.0.1:8086/v1/chat/completions \
  -H 'Authorization: Bearer local-llama' \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/mlx-one-vlm-request.json

python scripts/vlm_manual_smoke.py --mode http --model qwen2-vl-2b --repetitions 10
```

The HTTP smoke exercises authenticated discovery, completion, multi-turn image
history, streaming, one early disconnect and subsequent requests. Model IDs must
match `/v1/models`; the alias above is `qwen2-vl-2b`. To serve another target,
replace the model directory and alias, then use that alias in HTTP requests.
The API accepts at most four images per chat, 32 MiB multimodal request bodies,
20 MiB encoded per image, 20 million decoded pixels per image and a bounded
processing budget. Python/CLI allow up to eight images within their aggregate
pixel/token/memory limits. URLs pointing to local files and remote HTTP(S) images
are rejected; use base64 data URLs. Video and animated-image inputs are rejected.

## Benchmarks, result validation and README update

Stop the server before direct-model benchmarks. Each invocation below starts a
fresh process, verifies local checkpoint hashes, performs one warmup and five
measured repetitions, and writes actual output IDs and synchronized metrics.
No command downloads a model. Full image/phase/memory/provenance records stay in
JSON. The README publishes S1 separately from text benchmarks and other VLM
workloads. See [methodology](benchmarks/vlm/README.md).

```bash
python -m benchmarks.vlm.run \
  --checkpoint ~/mlx-one-models/vlm/qwen2-vl-2b \
  --model-id mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --workload VLM-S1 --output /tmp/V01-VLM-S1-measured.json
```

```bash
python -m benchmarks.vlm.run \
  --checkpoint ~/mlx-one-models/vlm/qwen2.5-vl-3b \
  --model-id mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --workload VLM-S1 --output /tmp/V02-VLM-S1-measured.json
```

```bash
python -m benchmarks.vlm.run \
  --checkpoint ~/mlx-one-models/vlm/qwen3.5-0.8b \
  --model-id mlx-community/Qwen3.5-0.8B-4bit \
  --workload VLM-S1 --output /tmp/V03-VLM-S1-measured.json
```

```bash
python -m benchmarks.vlm.run \
  --checkpoint ~/mlx-one-models/vlm/lfm2-vl-450m \
  --model-id mlx-community/LFM2-VL-450M-8bit \
  --workload VLM-S1 --output /tmp/V04-VLM-S1-measured.json
```

```bash
python -m benchmarks.vlm.run \
  --checkpoint ~/mlx-one-models/vlm/lfm2.5-vl-3b \
  --model-id LiquidAI/LFM2.5-VL-3B-MLX-4bit \
  --workload VLM-S1 --output /tmp/V05-VLM-S1-measured.json
```

For V01's other workloads, each loop iteration still launches a separate process:

```bash
for workload in VLM-S2 VLM-S3 VLM-S4; do
  python -m benchmarks.vlm.run \
    --checkpoint ~/mlx-one-models/vlm/qwen2-vl-2b \
    --model-id mlx-community/Qwen2-VL-2B-Instruct-4bit \
    --workload "$workload" --output "/tmp/V01-$workload-measured.json"
done
```

Inspect every answer and raw record. After a successful real measurement, copy
only that result over its `NOT_RUN` record and regenerate:

```bash
cp /tmp/V01-VLM-S1-measured.json docs/benchmarks/vlm/results/V01-VLM-S1.json
python -m benchmarks.vlm.readme_generator
python -m benchmarks.vlm.readme_generator --check
```

Use a new `/tmp/` output filename for another run; do not merge samples from
incompatible protocols or hardware. Unsupported/failed workloads remain pending
with a documented reason, rather than a fabricated zero or speed estimate.

## Manual checkpoint acceptance checklist

For each pinned checkpoint separately:

1. Download succeeds; all exact tensor, quantization and processor assets load.
2. Python and CLI generate sensible image-dependent text, not unresolved markers.
3. HTTP non-streaming and SSE requests work with the selected alias/key.
4. Two turns preserve image/text ordering; a changed image changes relevant answers.
5. Early stream close/disconnect cleans request state; the next request succeeds.
6. Repeated-request active memory reaches a bounded plateau; inspect the smoke's trace.
7. Run supported S1/S2/S3/S4 workloads, retaining actual counts and raw results.
8. Review semantic/OCR output; speed alone is not a functional quality result.
9. Mark `CHECKPOINT_SMOKE_PASSED` only after the exact checkpoint Python/CLI/API
   smokes pass. Promote to `EXPERIMENTAL_SUPPORTED` only with required functional
   regressions and measured evidence for any advertised performance.

Static images only: no video, audio, training, VLM tool calls, speculative decode,
managed image APC/paged KV or long-context qualification. LFM2.5-VL 3B is capped
at 32,768 context tokens despite its larger text-config maximum. All visual and
text framing tokens plus requested output must fit the effective context and
processing budget. See each model guide for architecture-specific limits.
