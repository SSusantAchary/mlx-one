# Qwen3-ASR and Qwen3-ForcedAligner

Native MLX inference for the original publisher checkpoints. ASR and alignment
are distinct tasks; the aligner cannot transcribe unknown speech. The implementation
has Python/CLI integration, but **checkpoint qualification is incomplete**.

## Model Architecture

```text
mono PCM, declared 16 kHz
         │
128-bin log-Mel (400 FFT / 160 hop; dynamic length, no 30-second trimming)
         │
100-frame chunks → three stride-2 Conv2D + GELU → flatten → projection
         │
per-chunk sinusoidal positions → audio Transformer → LayerNorm
         │
GELU projection → decoder-width audio embeddings
         │
replace exact <|audio_pad|> placeholders in checkpoint prompt
         │
Qwen3 causal decoder (Q/K RMSNorm, GQA, RoPE, SwiGLU)
         ├── ASR: tied vocabulary head → greedy autoregressive text
         └── ForcedAligner: 5,000-way head at <timestamp> tokens
                                → 80-ms classes → ordered word/character spans
```

Paper: [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337).
Reference source: [QwenLM/Qwen3-ASR, pinned commit](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff).
The reference's non-FlashAttention **SDPA** encoder uses global attention; its
FlashAttention path uses windows. This port reproduces the pinned SDPA path,
not FlashAttention output parity. All three audio positional axes share the
same indices, so its interleaved MRoPE reduces to ordinary nontraditional RoPE.

| Checkpoint | Registry / task | Architecture | Qualification |
| --- | --- | --- | --- |
| [ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `qwen3_asr` / transcribe | 18 audio layers, width 896; 28 text layers, width 1024 | English offline: 100 clips / 16 anchors pass; French offline: 25 clips pass; streaming diagnostic fails exact text |
| [ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `qwen3_asr` / transcribe | 24 audio layers, width 1024; 28 text layers, width 2048 | English offline 100 clips: WER/CER match reference, but FAIL at 15/16 exact anchors; other languages/streaming pending |
| [ForcedAligner 0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | `qwen3_forced_aligner` / align | 24 audio layers, width 1024; 28 text layers, width 1024; timestamp head | FAIL: 100 English clips, maximum boundary difference 800 ms exceeds 80-ms gate |

Full checkpoint pins, tensor names/shapes, precision and asset checksums:
[0.6B contract](../../../../../docs/qualification/archived-evidence.md),
[1.7B contract](../../../../../docs/qualification/archived-evidence.md),
[aligner contract](../../../../../docs/qualification/archived-evidence.md).
All stored weights are BF16; no automatic quantization or precision reduction.
Float32 component parity alone does not qualify these checkpoints.

The latest [closure investigation](../../../../../docs/audio-qualification-m4-32gb.md#stage-three-closure-investigation--rejected-numerical-candidate)
tested a rounding candidate against real checkpoints. It failed exact-transcript
and timestamp gates, so it remains opt-in benchmark tooling, **not** the loader's
default. Historical passing scopes and production behavior are preserved;
stage-three qualification is still **WIP**. As of 2026-10-04, these failures
no longer block Stage 4–6 development; they still prevent qualification claims.
The completed 1.7B 100-clip English run and isolated decoder/classifier diagnostics
are recorded in the [follow-up report](../../../../../docs/audio-qualification-m4-32gb.md#stage-three-follow-up--decoder-ties-and-isolated-timestamp-classifier).

## Installation and assets

Install mlx-one on Apple Silicon/macOS and activate `~/venvs/mlx` for native runs.
FFmpeg is needed only for file inputs; mono waveform arrays require explicit
`sample_rate=16000`. Other waveform rates and stereo arrays are rejected: resample
and downmix explicitly. File decoding produces mono 16-kHz PCM. The pinned
reference uses librosa for file resampling; cross-resampler parity is not qualified.

Checkpoints need `config.json`, `preprocessor_config.json`, `generation_config.json`,
`vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.json`, and the
publisher's standard safetensors layout (indexed two-shard export for 1.7B).
The loader validates names/shapes, shard indices and tied heads, selects one
authoritative safetensors layout, rejects ambiguous exports, and never executes
Hub code or restores pickle. Use `offline=True` only
after all pinned assets have been cached, or pass a complete local directory.

## Python and CLI

```python
from mlx_one.audio import QwenASROptions, load_audio_model, transcribe

model = load_audio_model(
    "Qwen/Qwen3-ASR-0.6B",
    revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
)
result = transcribe(model, "speech.wav", language="English",
                    options=QwenASROptions(context="Names: Alice, Berlin"))
print(result.text)
```

Omit `language` for checkpoint language detection. Use publisher language **names**
(English, French, German, Chinese), not Whisper codes. Context is the checkpoint's
system context, not Whisper initial-prompt tokens. The checkpoint chat template
is rendered verbatim, including its literal backslash-newline markers. There is
no text-chat, translation, timestamp inference, confidence or Whisper fallback.
Unavailable metrics remain `None` in the separate `SpeechTranscript` result.

```bash
mlx-one transcribe Qwen/Qwen3-ASR-0.6B speech.wav --language English \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 --json-output
mlx-one align Qwen/Qwen3-ForcedAligner-0.6B speech.wav \
  --text-file transcript.txt --language English --output alignment.json \
  --revision c7cbfc2048c462b0d63a45797104fc9db3ad62b7
```

```python
from mlx_one.audio import align_audio

result = align_audio(
    "Qwen/Qwen3-ForcedAligner-0.6B", "speech.wav", "Known transcript",
    language="English", revision="c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
)
for unit in result.units:
    print(unit.text, unit.start, unit.end)
```

Alignment retains the original text separately from cleaned word/character units.
It strips punctuation, emits Chinese ideographs individually and repairs ordered
boundaries using the reference algorithm. Scores are timestamp classes, **not**
ASR confidence. The pinned wrapper's per-request alignment limit is 180 seconds;
longer input fails explicitly rather than splitting/truncating it. Empty text
returns no units; nonempty text cannot be aligned against empty audio. Japanese
requires optional `nagisa==0.2.11`; Korean requires optional `soynlp==0.0.493`
and the pinned `korean_dict_jieba.dict` beside the checkpoint. Missing assets fail
explicitly; an untrained tokenizer is never substituted.

## Streaming and resource ownership

```python
from mlx_one.audio import open_audio_stream

with open_audio_stream(model, sample_rate=16000, language="English") as session:
    for pcm in ([0.0] * 16000, [0.0] * 16000):
        for event in session.push(pcm):
            print(event.kind, event.text)  # complete current hypothesis
    print(session.finish().text)
    session.reset()  # required before another utterance
```

The arrays above demonstrate lifecycle, not a speech-quality fixture. Streaming
reprocesses all accumulated audio: two-second chunks, two initial unfixed chunks,
then five-token rollback. Final-tail rollback follows the reference's distinct
finish behavior. It does **not** retain a causal audio-encoder cache. Events carry
complete hypotheses that may replace earlier text; do not concatenate them.
Use floating PCM samples, or explicitly typed `int16` arrays (scaled by 32,768).
Encoded bytes and stereo arrays are rejected.

```bash
mlx-one transcribe Qwen/Qwen3-ASR-0.6B short-utterance.wav \
  --stream --language English --json-output --offline \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0
```

Streaming NDJSON emits `partial` and one `final` event with sequence, audio
duration, model/revision and provenance. The explicit limit is **30 seconds per
utterance**; longer input raises an error, with no automatic segmentation.
`finish()` flushes pending PCM and rejects further input until `reset()`.
Context exit/cancellation clears only session-owned input/state and decoder
processes. Other sessions' model references and shared MLX allocator remain intact.
Audio caches are independent of text dense/APC/block caches.

## Limits, troubleshooting and evidence

Single-request inference is batch-size one. The 24-GiB process gate, dense audio
attention preflight, positional-context gate and token budget fail explicitly;
no hidden precision reduction, audio dropping, transcript truncation or CPU/
reference fallback. Longer offline requests may exceed the dense-attention budget.
Unsupported Whisper options are rejected. Existing exports are never overwritten.

See [audio qualification](../../../../../docs/audio-qualification-m4-32gb.md),
[network-free tests](../../../../../tests/test_qwen_audio.py),
[component parity harness](../../../../../benchmarks/qwen_audio_component_parity.py),
and [checkpoint qualification harness](../../../../../benchmarks/qwen_audio_qualification.py).
Reference outputs must be persisted in the isolated Transformers 4.57.6/PyTorch
2.7.1 environment before native execution. Failed gates remain unqualified;
Other language and streaming gates are not implied by architecture tests.
The checksummed FLEURS selections are recorded in the qualification report; fixture
preparation alone does not imply passing inference. Validation of streaming uses
the upstream rollback wrapper with a pinned Torch SDPA decoder, not vLLM-kernel parity.

Code and publisher checkpoints are Apache-2.0; retain the adjacent [NOTICE](NOTICE).
This independent MLX port is not an endorsement from Alibaba/Qwen.


## Native managed audio API (2026-10-24)

The common `AudioEngine` now exposes the implemented tasks through the existing
inference actor/model pool, grouped `mlx-one audio` commands, and native HTTP audio
routes. Encoded bytes, bounded PCM mixing/resampling, repeated requests and
cancellation use the common input/lifecycle contracts. See the
[exact manual download and test runbook](../../../../../docs/AUDIO_TESTING.md).
Existing checkpoint-specific evidence above remains historical and is not a new
standardized benchmark or an exhaustive numerical admission requirement.

The MLX community ASR 0.6B/1.7B and ForcedAligner 0.6B Q8 exports have explicit
native packed-weight loading, distinct from the original `thinker.*` publisher
layout. The loader preserves BF16 encoder exceptions and uses the shared affine
quantization support; unsupported formats fail explicitly. Pinned config/tensor
headers and native tiny Q8 execution were tested; these exact full checkpoints
were not downloaded/run. Historical 1.7B punctuation and aligner timestamp
reference disagreements remain unresolved. Alignment reports out-of-clip spans
as unmatched words rather than inventing confidence or clamping predictions.
