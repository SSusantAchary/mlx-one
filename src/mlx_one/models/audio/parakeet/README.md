# Parakeet TDT v2 / v3

Native MLX **offline** automatic speech recognition, registered as
`parakeet_tdt`. This is not RNNT-only Nemotron, an alignment model, or a streaming
API. Both checkpoints have independent contracts; loading one never substitutes
the other. Full qualification remains **WIP**. See the scoped
[M4 evidence](../../../../../docs/audio-qualification-m4-32gb.md).

## Model Architecture

```text
Mono PCM (16 kHz, declared sample rate)
                │
Preemphasis → centered STFT → 128 Slaney Mel filters → log → feature normalization
                │
Three convolutional subsampling stages (8× time reduction)
                │
24-layer FastConformer (1024-wide, 8-head relative attention,
half-step feed-forward blocks + depthwise convolution + normalization)
                │                         previous token + recurrent state
Encoder projection (1024 → 640)                          │
                │                  embedding → two-layer LSTM → projection
                └───────────────────────┬────────────────┘
                                  ReLU joint network
                                        │
                     token/blank logits + duration logits (0–4 frames)
                                        │
                      greedy TDT decoding → text + token timestamp records
```

Papers: [Fast Conformer with Linearly Scalable Attention](https://arxiv.org/abs/2305.05084)
and [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795).
Implemented dimensions come from the frozen checkpoints, not a universal
parameterization of every Parakeet release.

## Checkpoints and qualification

| Checkpoint | Frozen revision | Workflow / limits |
| --- | --- | --- |
| [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/tree/ae9ad07059c7c739ffaf932226a8fe64ae2620b0) | `ae9ad07059c7c739ffaf932226a8fe64ae2620b0` | English; requires isolated conversion of the original NeMo archive |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b) | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | English 100 clips / 16 anchors, French 25 and German 25 pass scoped offline gates; full qualification WIP |

Float32 component comparison passes at `atol=rtol=1e-4`, including silent-input
normalization, encoder and recurrent predictor. v2's 100 English short clips and
16 exact anchors pass against the pinned **converted Transformers** reference.
This does not establish original NeMo inference parity, word-timestamp accuracy,
long-form quality or streaming support. v3 passes its independent English,
French and German offline transcript gates. Outstanding timestamp and broader
gates are recorded separately in the qualification report. No passing architecture test is a general
checkpoint qualification claim.

## Installation and checkpoint assets

Install mlx-one normally on an Apple Silicon Metal host. `tokenizers` handles
text assets; neural inference and feature extraction are MLX. FFmpeg is used for
audio files. Python arrays require `sample_rate=16000`; other rates must be
resampled explicitly. No production PyTorch, Transformers or NeMo dependency.

Required files: `config.json`, `processor_config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, and canonical `model.safetensors`.
The loader validates all 723 tensor names, shapes and dtypes before model
materialization. Alternative/consolidated exports are not merged. The current
pinned checkpoints are float32, with integer BatchNorm counters; reduced-precision
and quantized variants are rejected.

v2 additionally requires `conversion-provenance.json` with verified asset hashes.
The converter retains original frontend buffers and configuration as safe assets;
original NeMo frontend/decoding parity remains a separate WIP gate. It preserves
v2's 1024 lexical IDs and final blank ID, rather than copying v3's vocabulary.

Run conversion only in an isolated validation environment containing PyTorch
2.7.1, safetensors, PyYAML, sentencepiece and protobuf, plus the official
Transformers checkout at `469230357aab0f2b303b0d638c1f8d06edb14184`:

```bash
python benchmarks/parakeet_conversion.py \
  --archive /checkpoints/parakeet-tdt-0.6b-v2.nemo \
  --reference-source /validation/transformers \
  --output /checkpoints/parakeet-v2-native
```

Only the frozen archive SHA-256
`d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`
is accepted. Unsafe paths, links, special files, duplicate members and excessive
extraction sizes are rejected before extraction. Tensor restoration uses
`weights_only=True` in the isolated converter; production never restores NeMo or
pickle archives. A new output directory is mandatory.

## Input and examples

No chat template or text prompt is used: supply mono audio. v3 uses the checkpoint's
automatic language behavior, not language-ID conditioning; explicit v3 `language`
selection is rejected. v2 accepts English only. Missing confidence and detected
language are left unset.

```python
from mlx_one.audio import load_audio_model, transcribe

model = load_audio_model(
    "nvidia/parakeet-tdt-0.6b-v3",
    revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
)
result = transcribe(model, "speech.wav")
print(result.text)
print(result.to_dict())  # SpeechTranscript, not Whisper JSON fields
```

```bash
mlx-one transcribe nvidia/parakeet-tdt-0.6b-v3 speech.wav \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b --json-output
mlx-one transcribe /checkpoints/parakeet-v2-native speech.wav --offline --json-output
```

For waveform arrays use `transcribe(model, pcm, sample_rate=16000)`. Empty arrays
return empty transcripts. Nonempty input shorter than 320 samples is rejected
because the pinned feature normalization requires at least two valid frames.
Nonfinite and non-mono arrays are rejected. File decoding uses FFmpeg's mono
16-kHz output, not Whisper's spectrogram.

## Settings, cache behavior and troubleshooting

- Greedy, batch-size-one TDT only. Duration-head classes are separate from lexical
  tokens. Token timestamps use the 80-ms encoder tick; these are **not word
  alignment**. `word_timestamps=True` is rejected until its qualification exists.
- `SpeechTranscript` preserves tokens, optional token timestamps and checkpoint
  provenance. Available language/confidence must not be inferred from Whisper fields.
- Streaming, translation, synthesis, known-text alignment and Qwen/Whisper-specific
  decoder options are explicitly unsupported. Do not use `--stream` for Parakeet.
- Predictor state is request-local. There are no text dense/APC/block cache handles
  and no global allocator clearing on cleanup. Cancellation checks occur between
  encoder layers and decoding steps.
- The dense encoder is not a bounded long-form streaming encoder. Resource checks
  enforce the 24-GiB process budget and reject excessively large attention requests.
  A token/resource limit raises an error; it never silently truncates audio/text.
- Missing v2 assets: run the isolated converter; the remote original archive is not
  directly loadable. Wrong vocab/precision/configuration: use the frozen checkpoint,
  not an alternate model. Metal failures: use native execution outside a sandbox
  that cannot initialize the GPU.

## Validation, source and license

Tests: [network-free contracts and archive safety](../../../../../tests/test_parakeet.py),
[Metal components and request isolation](../../../../../tests/test_parakeet_mlx.py).
Tools: [checkpoint contracts](../../../../../benchmarks/audio_checkpoint_contract.py),
[component comparison](../../../../../benchmarks/parakeet_component_parity.py),
[offline corpus comparison](../../../../../benchmarks/parakeet_qualification.py).
The frozen [official Transformers source](https://github.com/huggingface/transformers/tree/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/parakeet)
is validation-only; production has no reference fallback.

Publisher checkpoint terms are **CC-BY-4.0**, separate from mlx-one's Apache-2.0.
Retain attribution and review the exact model cards. See [NOTICE](NOTICE).


## Native managed audio API (2026-10-24)

The common `AudioEngine` now exposes the implemented tasks through the existing
inference actor/model pool, grouped `mlx-one audio` commands, and native HTTP audio
routes. Encoded bytes, bounded PCM mixing/resampling, repeated requests and
cancellation use the common input/lifecycle contracts. See the
[exact manual download and test runbook](../../../../../docs/AUDIO_TESTING.md).
Existing checkpoint-specific evidence above remains historical and is not a new
standardized benchmark or an exhaustive numerical admission requirement.

`mlx-community/parakeet-tdt-0.6b-v3` has an explicit MLX export bridge for its
NeMo-shaped config, native convolution layout, combined LSTM biases and vocabulary.
It maps to the existing FastConformer/TDT implementation, not CTC. New source
contract tests do not replace the historical NVIDIA-layout evidence or validate
word alignment. Token timestamps follow actual TDT durations on an 80-ms tick;
last-frame quantization may extend up to one tick beyond clip duration. Original
v2 conversion remains isolated tooling; original NeMo parity is still WIP.
