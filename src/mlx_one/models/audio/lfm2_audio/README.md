# LFM2.5 Audio

[Model guide index](../../../../../README.md#model-specific-documentation)

Native source support now includes the **MLX community Q4 export**, a strict task
loader, the shared MLX-One LFM2 hybrid backbone, FastConformer speech input,
audio adapter, sequential codebook prediction, and the separately supplied LFM
waveform detokenizer. Tasks are ASR, TTS, and interleaved speech-to-speech, with
ordered text/PCM streaming. Status: **NATIVE_IMPLEMENTED**, with experimental, input-dependent speech-to-speech
support. Real Q4 checkpoint transcription and TTS, plus a separate short spoken
question, have been exercised on Apple M4; see the
[hardware report](../../../../../docs/validation/audio-hf-2026-10-24/report.md)
and [measured performance](../../../../../docs/benchmarks/audio/README.md).
These checks do not establish corpus accuracy or human-rated speech quality.
On the primary 10-second audiobook fixture, greedy STS exceeds the 512-frame
(40.96-second) output bound. A publisher-style sampled diagnostic ends but has
unstable waveform amplitude. The separate 2.04-second question completes with a
normal-amplitude response; that result does not resolve the longer-input failure.

Primary checkpoint:
[`mlx-community/LFM2.5-Audio-1.5B-4bit`](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-4bit/tree/55d4de65fa10fa935561aaaa995271d8b8ff2ee0),
revision `55d4de65fa10fa935561aaaa995271d8b8ff2ee0`.
[LiquidAI's model](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) and
[official task recipes](https://github.com/Liquid4All/liquid-audio/tree/19e65845923a7f136442c95137884ec61eb386aa)
are references; the original publisher tensor layout is not silently treated as
the primary MLX export.

```text
16 kHz mono PCM → publisher log-Mel frontend → FastConformer → audio adapter
                                                                     ↓
Text embeddings → existing native LFM2 hybrid backbone ← speech features
                               ↓
            text logits / sequential Depthformer codebooks
                               ↓
  eight discrete codebooks → separate LFM waveform detokenizer → 24 kHz PCM
```

The publisher schema (`lfm`, `encoder`, `depthformer`, `preprocessor`) is parsed
separately from the old synthetic nested-config API. The latter remains available
for existing architecture tests; it is not used to reinterpret real checkpoint
weights. Model/codec tensor names and shapes were checked against pinned remote
JSON headers without downloading weights. Packed Q4/Q8 affine linear/embedding
weights remain packed; BF16 Conformer/normalization/convolution exceptions and
the separate codec retain their actual precision. Unknown/missing/extra weights
fail strict loading. No whole-model FP16 dequantization or external inference
runtime is used.

Input uses 16-kHz features; generated audio uses 24 kHz. ASR uses `Perform ASR.`,
TTS uses the four US/UK male/female publisher prompts, and speech-to-speech uses
`Respond with interleaved text and audio.`. Decoder state belongs to the request.
The codec excludes EOS frames and reconstructs a real waveform using the loaded
ISTFT window and overlap normalization. Mimi encoding, speaker cloning, speed
control, arbitrary OpenAI voices and multilingual/translation/timestamp guarantees
are not supported.

Streaming keeps native language-model state. The codec recomputes a bounded
prefix and withholds the unstable overlap-add tail; it is not a cached incremental
codec implementation. Output is bounded to 512 frames / 40.96 seconds, and token
or resource exhaustion fails explicitly. Cancellation and generator close release
request-owned caches without clearing another session's allocator state.

```python
from mlx_one.audio import AudioEngine

with AudioEngine("~/mlx-one-models/lfm2.5-audio-q4", offline=True) as model:
    transcript = model.transcribe("speech.wav")
    speech = model.generate_speech("Hello from MLX-One.", voice="uk_female")
    speech.save("generated.wav")
    response = model.speech_to_speech("question.wav")
    response.save("response.wav")
```

See the [download/testing runbook](../../../../../docs/AUDIO_TESTING.md),
[benchmark methodology](../../../../../docs/benchmarks/audio/README.md),
[native task tests](../../../../../tests/test_audio_pipeline_native.py),
[implementation report](../../../../../docs/validation/audio-native-2026-10-24/report.md),
and [source attribution](UPSTREAM.md). The adjacent [MIT license](UPSTREAM_LICENSE)
covers adapted reference code; publisher checkpoint terms remain separate.
