# Whisper

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `whisper`. The native encoder-decoder stack includes safe checkpoint
loading, log-Mel preprocessing, BPE, language detection, beam/sampled decoding,
segment and word timestamps, and WER/CER evaluation. Configurations cover tiny,
base, small, medium, large-v3, and large-v3-turbo variants.

The root README reports passing pinned tiny/turbo smoke gates. The
[integration test](../../../../../tests/test_native_whisper_integration.py) pins:

| Checkpoint | Revision |
| --- | --- |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | `60be3615a4d667e1258e8ad29130467587c489aa` |

These smoke contracts do not establish general transcription quality, WER,
timestamp accuracy, or throughput for all variants. The family remains a candidate
for broader task/hardware qualification.

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
