# Moonshine Tiny / Base

Registry type: `moonshine`. Native MLX offline English transcription for the
original Tiny/Base checkpoints, not the newer Moonshine streaming architectures.
Python and CLI are available; there is no new HTTP or UI integration.

## Model Architecture

Paper: [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608).
The implemented architecture consumes raw waveforms, not Whisper Mel features:

```text
Mono float32 waveform [batch, samples], 16 kHz
                 |
       Conv1d k=127, stride=64, no bias -> tanh
                 |
       GroupNorm (one group, eps=1e-5)
                 |
       Conv1d k=7, stride=3 -> GELU -> Conv1d k=3, stride=2 -> GELU
                 |
       Encoder layers: pre-LayerNorm -> self-attention + partial interleaved RoPE
                       -> residual -> pre-LayerNorm -> GELU MLP -> residual
                 |
       Final LayerNorm -> encoder features / immutable cross-attention K/V
                 |
BOS / generated tokens -> tied embedding -> cached causal decoder self-attention
                 -> audio cross-attention -> SiLU-gated MLP -> final LayerNorm
                 -> tied vocabulary projection -> greedy token -> EOS
```

| Setting | Tiny | Base |
| --- | ---: | ---: |
| Advertised parameters | 27M | 61M |
| Hidden width | 288 | 416 |
| Encoder / decoder layers | 6 / 6 | 8 / 8 |
| Attention heads | 8 | 8 |
| Feed-forward width | 1152 | 1664 |
| Partial rotary fraction | 0.9 | 0.62 |
| Actual rotary dimension per head | 32 | 32 |
| Vocabulary | 32768 | 32768 |

The native attention calculation does not need the reference's zero head padding;
it retains the original unpadded head-dimension scaling. Both head widths and
rotary fractions pass deterministic float32 component checks, including cached
versus full-sequence decoding. No architectural settings are changed to fit memory.

## Checkpoints and qualification

| Checkpoint | Pinned revision | Measured status |
| --- | --- | --- |
| [moonshine-ai/moonshine-tiny](https://huggingface.co/moonshine-ai/moonshine-tiny) | `390624ed33d594443aa4aa221f5b9f283b545b5a` | M4 English short-form gates pass |
| [moonshine-ai/moonshine-base](https://huggingface.co/moonshine-ai/moonshine-base) | `7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf` | M4 English short-form gates pass |

Each checkpoint was run on 100 fixed English LibriSpeech test-clean clips, with
16/16 exact greedy reference anchors, identical corpus WER/CER and zero active
MLX allocation growth across requests. Float32 components pass `atol=rtol=1e-4`.
See [measured evidence and workload limitations](../../../../../docs/audio-qualification-m4-32gb.md#moonshine-tinybase-stage-two-results).
These are narrow checkpoint/task/hardware passes, not long-form, live streaming,
noise robustness, diarization, alignment or general ASR quality qualification.

## Installation and assets

Install `mlx-one` on Apple Silicon with MLX available. `tokenizers`, safetensors
and Hugging Face Hub are already core dependencies; no production Torch,
Transformers or upstream inference package is needed. FFmpeg is required only
for file input. Files are decoded/downmixed and resampled to mono float32 16 kHz;
Python arrays must already have that rate and explicitly declare it.

A complete local checkpoint needs `config.json`, `generation_config.json`,
`preprocessor_config.json`, `tokenizer.json` and the selected safetensors layout.
The processor must declare raw waveform input with normalization disabled.
The loader rejects missing assets, incompatible processors, untied output
projections and unknown/shape-incompatible tensors. An index selects one shard
layout; it never merges duplicate exports or precision variants. Stored float32
checkpoint weights are preserved; there is no automatic quantization or fallback.

## CLI and Python

```bash
mlx-one transcribe moonshine-ai/moonshine-tiny recording.wav \
  --revision 390624ed33d594443aa4aa221f5b9f283b545b5a --json-output
# After downloading the complete snapshot, repeat with --offline.
mlx-one transcribe moonshine-ai/moonshine-base recording.flac \
  --revision 7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf --offline --json-output
```

Both commands inherit `--cache-dir`. Omit Whisper-specific beam, sampling,
fallback, seed and prompt flags: passing them explicitly to Moonshine is an
error. Translation and `--word-timestamps` are also unsupported.

```python
import numpy as np
from mlx_one.audio import ASRDecodeOptions, load_audio_model, transcribe

model = load_audio_model(
    "moonshine-ai/moonshine-tiny",
    revision="390624ed33d594443aa4aa221f5b9f283b545b5a",
    offline=True,
)
result = transcribe(model, "recording.wav")
print(result.text)

# An already decoded mono waveform requires its declared rate.
waveform = np.zeros(16000, dtype=np.float32)
result = transcribe(
    model, waveform, sample_rate=16000,
    options=ASRDecodeOptions(max_tokens=193),
)
print(result.to_json())
```

`SpeechTranscript` is separate from the legacy Whisper result. It includes token
IDs, duration, sample rate, checkpoint provenance and timings. Language is unset
unless the caller explicitly supplies `language="en"`; unavailable confidence
and timestamps are `None`, never fabricated Whisper scores. Silence can produce
hallucinations; there is no implicit voice-activity filter. The waveform example
demonstrates input shape, not a guarantee of empty text on silence.

## Limits and troubleshooting

- Empty arrays return an empty result without an encoder call. Nonempty inputs
  shorter than 895 samples are rejected rather than implicitly padded. Stereo,
  nonfinite arrays and undeclared/non-16-kHz array rates are rejected.
- Default decoding follows checkpoint `max_length=194`: BOS plus at most 193
  generated tokens. `ASRDecodeOptions.max_tokens` sets an explicit generation
  budget; exhausting it without EOS raises `AudioResourceLimitError`, not a
  silently truncated transcript. This is not an input cropping policy.
- The dense waveform encoder has quadratic attention memory. A conservative
  working-set preflight and runtime process checks enforce the 24 GiB budget.
  Long recordings can fail explicitly; automatic chunking is not implemented.
- Each request owns its decoder caches. Success, exceptions and cancellation
  release those caches without clearing the shared allocator or unloading models.
  No text dense/APC/block cache policy applies to this audio state.
- A snapshot with a different `model_type`, RoPE scaling, grouped-query
  configuration, quantization or altered generation processors is not this
  supported original checkpoint contract. Use a complete pinned snapshot.

## Tests, references and licenses

```bash
python -m pytest -q tests/test_moonshine_architecture.py
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest -q tests/test_moonshine_mlx.py
```

See [network-free contracts](../../../../../tests/test_moonshine_architecture.py),
[Metal execution/isolation tests](../../../../../tests/test_moonshine_mlx.py),
[component parity harness](../../../../../benchmarks/moonshine_component_parity.py),
[checkpoint qualification harness](../../../../../benchmarks/audio_qualification.py),
and [qualification report](../../../../../docs/audio-qualification-m4-32gb.md).
Reference: [Transformers Moonshine at the pinned source commit](https://github.com/huggingface/transformers/tree/cc832f9055ba11c8c55f918ab4bda9472b910d48/src/transformers/models/moonshine).
The validation implementation is isolated from production.

Publisher model cards identify the Tiny/Base checkpoints as MIT-licensed. This
is separate from mlx-one's Apache-2.0 license. The adapted Transformers
architecture reference is Apache-2.0; see [NOTICE](NOTICE). Respect publisher
warnings about hallucination, non-consensual recording and high-risk deployment.


## Native managed audio API (2026-10-24)

The common `AudioEngine` now exposes the implemented tasks through the existing
inference actor/model pool, grouped `mlx-one audio` commands, and native HTTP audio
routes. Encoded bytes, bounded PCM mixing/resampling, repeated requests and
cancellation use the common input/lifecycle contracts. See the
[exact manual download and test runbook](../../../../../docs/AUDIO_TESTING.md).
Existing checkpoint-specific evidence above remains historical and is not a new
standardized benchmark or an exhaustive numerical admission requirement.
