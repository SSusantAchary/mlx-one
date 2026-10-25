# Native audio: manual download and test runbook

Install on Apple Silicon with Metal:

```bash
python -m pip install -e '.[audio,server]'
brew install ffmpeg
```

Neural inference uses MLX-One's native modules. Runtime code does not require
mlx-audio, mlx-whisper, Transformers, PyTorch, NeMo, ONNX or llama.cpp. FFmpeg handles
encoded media, and NumPy handles ordinary PCM mixing/resampling. The optional
`audio` extra documents the workflow; its Python requirements are already core.

## Status and checkpoint selection

Native implementations and real-checkpoint results are tracked separately. The
primary catalog has now been exercised using pinned Hugging Face data on Apple M4;
see the [hardware test report](validation/audio-hf-2026-10-24/report.md) and
[audio performance coverage](benchmarks/audio/coverage.md). These short checks
establish scoped execution and measured speed, not corpus accuracy, alignment
accuracy or human-rated speech quality. The earlier source-only implementation
report remains historical. LFM speech-to-speech has input-dependent failures;
Qwen alignment retains its known timestamp discrepancy.

| Key | Checkpoint | Primary precision | Tasks / limits |
| --- | --- | --- | --- |
| A01 | openai/whisper-tiny | FP32 | ASR, language detection, translation, segment/word timestamps; timestamp quality requires manual review |
| A02 | openai/whisper-large-v3-turbo | FP16 | Native Whisper task protocol; broader language/translation quality not implied |
| A03 | distil-whisper/distil-large-v3 | FP16 | English distilled ASR; no multilingual/translation quality claim |
| A04 | UsefulSensors/moonshine-tiny | FP32 | English offline ASR; no language detection, timestamps, translation or streaming |
| A05 | UsefulSensors/moonshine-base | FP32 | Same task boundary as Tiny |
| A06 | mlx-community/Qwen3-ASR-0.6B-8bit | Q8 / BF16 | ASR, generated language label, bounded accumulated utterance |
| A07 | mlx-community/Qwen3-ASR-1.7B-8bit | Q8 / BF16 | Same source tasks; historical publisher BF16 punctuation discrepancy remains documented |
| A08 | nvidia/parakeet-tdt-0.6b-v2 | FP32 | Original NeMo archive requires isolated conversion; original NeMo parity remains WIP; English only |
| A09 | mlx-community/parakeet-tdt-0.6b-v3 | FP32 | MLX export mapping to existing native TDT; automatic multilingual behavior, no language-hint conditioning |
| A10 | mlx-community/LFM2.5-Audio-1.5B-4bit | Q4 / BF16 | Native ASR, no language-hint/translation/timestamp API |
| A11 | mlx-community/Qwen3-ForcedAligner-0.6B-8bit | Q8 / BF16 | Known-text alignment, not ASR; WIP timestamp discrepancy |
| A12 | Same LFM checkpoint as A10 | Q4 / BF16 | TTS, four publisher voice prompts, waveform detokenizer |
| A13 | Same LFM checkpoint as A10 | Q4 / BF16 | Experimental speech-to-speech; short question completes, primary 10-second input exceeds output bound |

Exact immutable revisions, assets and estimated bytes are in
[checkpoints.json](../benchmarks/audio/checkpoints.json). Download one checkpoint at
a time, test it, then keep/delete only the directory you explicitly created.
A10/A12/A13 share one download; do not download that checkpoint three times.
`mlx-community/whisper-tiny` exists but uses an incompatible minimal MLX export
without the tokenizer/processor assets required by this loader; use A01.
Original LiquidAI weights are a reference, not the primary supported MLX export.
Publisher Qwen and NVIDIA layouts retain their existing native loaders/evidence.

The [per-checkpoint command list](benchmarks/audio/manual-commands.md) supplies
exact download, CLI and benchmark invocations for every primary checkpoint/task.

## Explicit manual downloads

These commands download large files **only when you run them**:

```bash
python -m benchmarks.audio.download --key A06 --output ~/mlx-one-models/qwen3-asr-0.6b-q8
python -m benchmarks.audio.download --key A11 --output ~/mlx-one-models/qwen3-aligner-q8
python -m benchmarks.audio.download --key A10 --output ~/mlx-one-models/lfm2.5-audio-q4
```

Use any other key from the table with a new output directory. The downloader pins
the recorded Hub SHA, fetches selected safe assets, and writes hashes to
`mlx-one-download.json`. It refuses to overwrite an existing model directory.
The benchmark runner is local-only and rechecks that manifest; `--offline` APIs
never silently fetch missing files. Missing/incompatible assets fail explicitly.

For A08, download the pinned archive manually, then follow the existing converter
in [the Parakeet guide](../src/mlx_one/models/audio/parakeet/README.md). The actual
conversion command is:

```bash
python benchmarks/parakeet_conversion.py \
  --archive /path/to/parakeet-tdt-0.6b-v2.nemo \
  --reference-source /path/to/pinned-transformers \
  --output ~/mlx-one-models/parakeet-v2-native
```

This isolated conversion uses the frozen source/dependency versions and archive
hash in that guide; production inference never imports NeMo or restores pickle.
Use its generated provenance; do not relabel a v3 export as v2. Register the
verified converted directory for the standard benchmark runner with:

```bash
python -m benchmarks.audio.convert_manifest --model ~/mlx-one-models/parakeet-v2-native
```

This validates the pinned archive/conversion provenance and every converted asset
hash before writing `mlx-one-download.json`. The generic downloader does not
pretend it has converted an archive.

## Python

```python
from mlx_one.audio import AudioEngine, AudioInput

with AudioEngine("~/mlx-one-models/qwen3-asr-0.6b-q8", offline=True) as model:
    result = model.transcribe("speech.wav", language="English")
    print(result.text, result.language, result.duration)
    print(model.detect_language("speech.wav"))  # ASR-generated label, no fake probability
```

Pass `AudioInput(encoded_bytes, media_type="audio/wav")`, or floating / explicitly
`int16` waveforms with their `sample_rate`. `[samples, 2]` stereo is mixed to mono;
8–192 kHz waveforms are resampled with a band-limited sinc filter. Input amplitudes
are not normalized generically. Each family retains its own frontend. An `AudioInput(offset_seconds=...)` shifts
returned transcript/alignment spans onto that timeline; duration stays clip-local. New managed
calls bound input to 180 seconds/25 MiB encoded bytes; offline Whisper direct file
inference additionally supports bounded ten-minute input. Cancellation uses a
`threading.Event` passed as `cancel=`. Close the context/stream to release ownership.

```python
with AudioEngine("~/mlx-one-models/qwen3-aligner-q8", offline=True) as model:
    result = model.align("speech.wav", "Known transcript.", language="English")
    print(result.units, result.unmatched_words, result.provenance)

with AudioEngine("~/mlx-one-models/lfm2.5-audio-q4", offline=True) as model:
    result = model.generate_speech("Hello from MLX-One.", voice="uk_female")
    result.save("generated.wav")
    response = model.speech_to_speech("question.wav")
    response.save("response.wav")
    print(response.text)
```

WAV/PCM speech output is mono signed 16-bit at **24 kHz**; native model output is a
finite floating waveform. Output duration is derived from actual samples. Writes
refuse to replace existing files. LFM uses its own audio encoder/depthformer and
separate published waveform detokenizer. The Mimi tokenizer artifact is not needed
for these speech-input/frontend and generated-code/output-decoder tasks. Codec
encoding of arbitrary audio into Mimi codes, voice cloning, speed control and
OpenAI stock voices are not supported.

## Streaming and lifecycle

```python
with AudioEngine("~/mlx-one-models/qwen3-asr-0.6b-q8", offline=True) as model:
    for event in model.stream_transcribe(pcm_chunks, sample_rate=16000, language="English"):
        print(event.kind, event.text)  # complete replaceable hypothesis

with AudioEngine("~/mlx-one-models/lfm2.5-audio-q4", offline=True) as model:
    for event in model.stream_generate("Hello from native MLX."):
        if event.kind == "audio_chunk":
            consume_pcm(event.waveform, event.sample_rate, event.sample_offset)
        elif event.kind == "text_chunk":
            print(event.text, end="")
        elif event.kind == "completion":
            print(event.timings)
```

Qwen accepts finite mono PCM and bounds each accumulated utterance to 30 seconds.
Chunks deterministically trigger two-second decoding boundaries; text may be
revised, so do not concatenate hypotheses. There is one final result, followed by
completion in the HTTP framing. This is not a causal/incremental encoder cache.
A new call owns a new session; direct sessions also offer explicit `reset()`.

LFM preserves language-model state, alternates text/audio in publisher order and
emits only decoded PCM. The codec currently recomputes a **bounded prefix**; it
does not retain a streaming codec KV cache. Overlapping ISTFT tail samples are
withheld until stable or final flush, and EOS codes never reach the codec. The
output bound is 512 audio frames (40.96 seconds); a generation-token or frame
limit raises an error instead of returning silently truncated speech. Buffered
request/cache/codec state is released on success, error, cancellation and close.
Audio uses the existing actor/device executor/pool, not a second model manager.
`AudioEngine` loads checkpoint paths on its device-owner thread; custom loaders
must construct/evaluate their lazy MLX arrays on that owner.

## CLI

```bash
mlx-one audio transcribe --model ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --audio speech.wav --language English --output transcript.json --offline
mlx-one audio detect-language --model ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --audio speech.wav --offline
mlx-one audio transcribe --model ~/mlx-one-models/whisper-tiny \
  --audio speech.wav --word-timestamps --offline
mlx-one audio transcribe --model ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --audio short-utterance.wav --language English --stream --offline
mlx-one audio align --model ~/mlx-one-models/qwen3-aligner-q8 \
  --audio speech.wav --text "Known transcript." --language English --offline
mlx-one audio speak --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --text "Hello from MLX-One." --voice uk_female --output generated.wav --offline
mlx-one audio speak --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --audio question.wav --output response.wav --offline
```

The existing top-level `mlx-one transcribe` and `mlx-one align` remain available.
Their model/input arguments are positional; the new grouped commands use the
options above. Do not copy options between these two interfaces blindly.

## HTTP

One startup audio model, existing server, dedicated port and explicit alias:

```bash
mlx-one serve ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --offline --host 127.0.0.1 --port 8085 --alias local-asr --no-warmup

curl -sS http://127.0.0.1:8085/v1/audio/transcriptions \
  -F model=local-asr -F file=@speech.wav -F language=English -F response_format=verbose_json
curl -sS http://127.0.0.1:8085/v1/audio/language \
  -F model=local-asr -F file=@speech.wav
curl -N http://127.0.0.1:8085/v1/audio/transcriptions \
  -F model=local-asr -F file=@short-utterance.wav -F stream=true
```

Streaming transcription uses SSE with complete `partial`/`final` hypotheses,
then `completion`, or a terminal `error`. The upload is buffered and bounded;
this endpoint is not a realtime bidirectional protocol. Whisper translation uses
`/v1/audio/translations`; JSON/text/verbose JSON and requested word/segment
`timestamp_granularities[]` are supported only for applicable Whisper checkpoints.

Restart with the aligner and alias `local-align` for:

```bash
curl -sS http://127.0.0.1:8085/v1/audio/alignments \
  -F model=local-align -F file=@speech.wav -F text="Known transcript." -F language=English
```

Restart with LFM and alias `local-audio` for:

```bash
mlx-one serve ~/mlx-one-models/lfm2.5-audio-q4 \
  --offline --host 127.0.0.1 --port 8085 --alias local-audio --no-warmup
curl -sS http://127.0.0.1:8085/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-audio","input":"Hello from MLX-One.","voice":"uk_female","response_format":"wav"}' \
  --output http-generated.wav
curl -sS http://127.0.0.1:8085/v1/audio/speech-to-speech \
  -F model=local-audio -F file=@question.wav --output http-response.wav
```

Add `--api-key your-key` to the server and `-H 'Authorization: Bearer your-key'` to
each request when authentication is wanted. The API model must equal the loaded
alias. Unsupported tasks/voices/formats/speed fail explicitly. No OpenAI realtime
WebSocket compatibility is claimed. Generated speech with `stream: true` returns
NDJSON frames (`kind`, sequence, sample offset/rate, text, base64 signed-16 PCM),
then completion/error; it is not a raw WAV stream or an SSE text delta.

## Manual checkpoint smoke and benchmarks

First run two requests and review transcript/word boundaries or listen to both
waveforms. Confirm no active model leases remain between requests or after close:

```bash
python scripts/audio_manual_smoke.py --model ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --audio speech.wav --language English --output /tmp/qwen-audio-smoke
python scripts/audio_manual_smoke.py --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --task synthesize --text-file tts-30-words.txt --output /tmp/lfm-audio-smoke
```

Prepare fixed clips and known transcripts under a local fixture directory. Pin
source/license/hashes in a manifest; use EN/FR/DE for multilingual Qwen/v3 Parakeet,
silent/noisy/leading/trailing-silence samples, repeated words and punctuation,
short and long input, and multiple utterances. The implementation's tiny fixtures
are intentionally engineered control-flow tests, not speech-quality substitutes.

```bash
python -m benchmarks.audio.run \
  --model ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --checkpoint mlx-community/Qwen3-ASR-0.6B-8bit \
  --revision 89e96d92ba34aca20b3e29fb10cc284097d1219f --precision 'Q8 / BF16' \
  --workload ASR-S1 --audio fixtures/english-10s.wav --text-file fixtures/english-10s.txt \
  --language English --output docs/benchmarks/audio/records/A06-ASR-S1-measured.json

python -m benchmarks.audio.run \
  --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --checkpoint mlx-community/LFM2.5-Audio-1.5B-4bit \
  --revision 55d4de65fa10fa935561aaaa995271d8b8ff2ee0 --precision 'Q4 / BF16' \
  --task synthesize --workload TTS-S1 --text-file tts-30-words.txt --voice uk_female \
  --output docs/benchmarks/audio/records/A12-TTS-S1-measured.json
```

Use the same runner for `ASR-S2/S3/S4`, `ASR-S5 --task stream_transcribe`,
`ALIGN-S1 --task align --boundaries-file known-boundaries.json`,
`TTS-S2 --task synthesize` and `STS-S1 --task speech-to-speech`. Boundary files are
ordered `{text,start,end}` lists in seconds. Keep each language/fixture/voice/type
in a distinct output JSON. `--measurement-type python-engine` and
`--measurement-type full-file-pipeline` are separate protocols, not interchangeable
rows. The direct Whisper ten-minute path is available for ASR-S3; managed input
and Qwen utterance bounds remain unchanged.

Replace the corresponding pending path in
[publication.json](benchmarks/audio/publication.json) with the measured JSON path,
then regenerate and validate the audio region of the README. Retain raw JSONs:

```bash
python -m benchmarks.audio.readme_generator --write
python -m benchmarks.audio.readme_generator --check
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest -q --no-cov \
  tests/test_audio_pipeline.py tests/test_audio_pipeline_native.py tests/test_audio_benchmark.py
```

Do not run or reinstate the retired exhaustive numerical qualification campaigns.
Historical Qwen investigations remain linked evidence, not a requirement to run
old reference environments before every new checkpoint admission.
