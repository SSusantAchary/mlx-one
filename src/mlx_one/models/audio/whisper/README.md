# Whisper

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `whisper`. The native encoder-decoder stack includes safe checkpoint
loading, log-Mel preprocessing, BPE, language detection, beam/sampled decoding,
segment and word timestamps, and WER/CER evaluation. Configurations cover tiny,
base, small, medium, large-v3, large-v3-turbo, and Distil-Whisper large-v3 variants.

The root README reports passing pinned tiny/turbo smoke gates. The
[integration test](../../../../../tests/test_native_whisper_integration.py) pins:

| Checkpoint | Revision |
| --- | --- |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | `60be3615a4d667e1258e8ad29130467587c489aa` |
| [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) | `8031d2e6ce6631b7fc45629dddfc00271116d981` |

These smoke contracts do not establish general transcription quality, WER,
timestamp accuracy, or throughput for all variants. The family remains a candidate
for broader task/hardware qualification.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Audio waveform → log-Mel features
                    ↓
┌──────────────────── Audio encoder ──────────────────────────┐
│ Conv1d + GELU → strided Conv1d + GELU → position embeddings │
│ Bidirectional Transformer encoder → final LayerNorm         │
└──────────────────────────┬──────────────────────────────────┘
                           │ cross-attention keys/values
Prefix / previous text tokens → token + position embeddings
                           ↓
┌──────────────────── Text decoder ───────────────────────────┐
│ Causal self-attention → audio cross-attention → MLP         │
│ Residuals + LayerNorm; self/cross-attention caches          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
         Tied vocabulary projection → text/task/timestamp tokens
```

## Papers and upstream references

Paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356).

Distillation reference: [Distil-Whisper: Robust Knowledge Distillation via
Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430), and the
[publisher's large-v3 model card](https://huggingface.co/distil-whisper/distil-large-v3).

Language/task/timestamp behavior is encoded in the decoder token protocol;
word-timestamp postprocessing is not a separate audio encoder.

## Distil-Whisper large-v3

This checkpoint keeps the 32-layer Whisper large-v3 encoder and uses a two-layer
decoder (width 1280, 20 heads, 128 Mel bins). It uses the same `whisper` registry
type and public transcription result as other Whisper variants, not a new ASR
architecture or upstream-library runtime bridge.

The [M4 qualification report](../../../../../docs/audio-qualification-m4-32gb.md)
records 100 English test-clean clips, 16 exact transcript anchors, matching
WER/CER, and 16 reference-matched segment-timestamp clips. This does not qualify
word timestamps, non-English quality, translation or general long-form workloads.

The loader chooses `model.safetensors` or its authoritative shard index and does
not combine the publisher's alternate `model.fp32.safetensors` export. Explicit
tied output projections must match decoder embeddings. Generation-file suppression
IDs and alignment heads are validated against the two-layer decoder.

```bash
mlx-one transcribe distil-whisper/distil-large-v3 recording.wav \
  --revision 8031d2e6ce6631b7fc45629dddfc00271116d981 \
  --language en --beam-size 1 --best-of 1 --temperature 0 --json-output
```

```python
from mlx_one.audio import WhisperDecodeOptions, load_audio_model, transcribe

model = load_audio_model(
    "distil-whisper/distil-large-v3",
    revision="8031d2e6ce6631b7fc45629dddfc00271116d981",
    offline=True,  # Requires a complete previously cached snapshot.
)
result = transcribe(
    model, "recording.wav", language="en",
    options=WhisperDecodeOptions(
        temperatures=(0.0,), beam_size=1, best_of=1,
        condition_on_previous_text=False, max_tokens=128,
        compression_ratio_threshold=None, logprob_threshold=None,
        no_speech_threshold=None,
    ),
)
print(result.text)
```

Historical default beam/fallback/previous-text behavior remains available.
For explicit text-only generation, Python decode options expose
`without_timestamps=True`; do not use that profile as segment-timestamp evidence.

## Installation and assets

Use Python 3.10+ and install `python -m pip install -e .` from the repository root.
Execution requires Apple Silicon/Metal. FFmpeg must be installed and on `PATH`
for audio-file input. Already-decoded mono 16-kHz samples bypass FFmpeg.

Keep `config.json`, safetensors weights/index, `preprocessor_config.json`,
`generation_config.json` when provided, `vocab.json`, `merges.txt`, tokenizer
configuration, and special/added token metadata. The processor's Mel-bin count
must match the checkpoint: tiny and turbo have different frontend dimensions.

## CLI and Python transcription

Supply your own `recording.wav`. These commands may download pinned model weights.

```bash
mlx-one inspect openai/whisper-tiny \
  --revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af
mlx-one transcribe openai/whisper-tiny recording.wav \
  --revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af \
  --language en --word-timestamps --json-output
```

```python
from mlx_one import WhisperDecodeOptions, transcribe

result = transcribe(
    "openai/whisper-tiny",
    "recording.wav",
    revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    language="en",
    word_timestamps=True,
    options=WhisperDecodeOptions(seed=0),
)
print(result.text)
for segment in result.segments:
    print(segment.start, segment.end, segment.text)
```

Use `language=None` for detection. `task="transcribe"` preserves the source
language; `task="translate"` selects the Whisper translation task. An
`initial_prompt` supplies transcript context, not a role-based chat conversation.

## Settings, cache, and troubleshooting

- Decode options include temperatures, beam size, best-of sampling, thresholds,
  and previous-text conditioning. Validate task accuracy on representative audio.
- Array inputs must already be mono 16-kHz samples; passing stereo arrays or a
  different sample rate does not request automatic resampling.
- A file-decode failure requires checking FFmpeg and the input format. A Mel-bin
  mismatch requires the checkpoint's own processor assets.
- Encoder-decoder attention state differs from causal text KV. Whisper does not
  inherit dense Qwen2/Qwen2.5 block KV/APC support.
- Use pinned `whisper-wer-v1`/`whisper-cer-v1` evaluation protocols for evidence;
  smoke outputs alone do not qualify ASR quality.

## Implementation and validation

See [configuration](config.py), [model](model.py), [loader](loading.py),
[processing](processing.py), [tokenizer](tokenizer.py), and
[transcription service](../../../audio/service.py).

```bash
pytest -q tests/test_native_whisper_architecture.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_whisper_mlx.py
MLX_ONE_RUN_WHISPER_INTEGRATION=1 pytest -q tests/test_native_whisper_integration.py
```

See [architecture tests](../../../../../tests/test_native_whisper_architecture.py),
[synthetic Metal tests](../../../../../tests/test_native_whisper_mlx.py), and
[qualification levels](../../../../../README.md#capability-levels). Integration
execution requires Metal, model storage, and sufficient host memory.


## Native managed audio API (2026-10-24)

The common `AudioEngine` now exposes the implemented tasks through the existing
inference actor/model pool, grouped `mlx-one audio` commands, and native HTTP audio
routes. Encoded bytes, bounded PCM mixing/resampling, repeated requests and
cancellation use the common input/lifecycle contracts. See the
[exact manual download and test runbook](../../../../../docs/AUDIO_TESTING.md).
Existing checkpoint-specific evidence above remains historical and is not a new
standardized benchmark or an exhaustive numerical admission requirement.

Word alignment now selects each segment's own generated-token attention rows and
its actual audio-frame range, excluding prefix/EOS rows and padded audio. This
fixes row/offset handling; manually reviewed word-timestamp accuracy remains
separate. Local file decoding defaults to a ten-minute/64-MiB bound; HTTP and the
managed API enforce tighter documented bounds. Translation and language detection
are available only where the checkpoint token/task protocol supports them.
