# Per-checkpoint manual commands

Run from the repository root after installing audio/server dependencies. These are
reusable commands. The subsequent Hugging Face hardware run is documented in the
[hardware report](../../validation/audio-hf-2026-10-24/report.md).
Use the [runbook](../../AUDIO_TESTING.md) for API/HTTP/streaming, fixture permissions,
conversion dependencies, resource limits and result publication. Each revision is immutable. Prepare `/tmp/mlx-one-audio-fixtures` with the command
in [the methodology](README.md#hugging-face-fixture-and-reproduction) first.
The primary STS-S1 input is retained as a documented failing case; do not publish
an incomplete/truncated generation as a successful benchmark.

## A01: Whisper Tiny — transcribe

Revision: `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. Native implementation; see the linked real-checkpoint hardware evidence.

```bash
python -m benchmarks.audio.download --key A01 --output ~/mlx-one-models/whisper-tiny
mlx-one audio transcribe --model ~/mlx-one-models/whisper-tiny \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --language en --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/whisper-tiny \
  --checkpoint openai/whisper-tiny \
  --revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af --precision 'FP32' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --language en \
  --output docs/benchmarks/audio/records/A01-ASR-S1-measured.json
```

## A02: Whisper Large v3 Turbo — transcribe

Revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Native implementation; see the linked real-checkpoint hardware evidence.

```bash
python -m benchmarks.audio.download --key A02 --output ~/mlx-one-models/whisper-turbo
mlx-one audio transcribe --model ~/mlx-one-models/whisper-turbo \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --language en --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/whisper-turbo \
  --checkpoint openai/whisper-large-v3-turbo \
  --revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 --precision 'FP16' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --language en \
  --output docs/benchmarks/audio/records/A02-ASR-S1-measured.json
```

## A03: Distil-Whisper Large v3 — transcribe

Revision: `8031d2e6ce6631b7fc45629dddfc00271116d981`. Native implementation; see the linked real-checkpoint hardware evidence.

```bash
python -m benchmarks.audio.download --key A03 --output ~/mlx-one-models/distil-large-v3
mlx-one audio transcribe --model ~/mlx-one-models/distil-large-v3 \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --language en --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/distil-large-v3 \
  --checkpoint distil-whisper/distil-large-v3 \
  --revision 8031d2e6ce6631b7fc45629dddfc00271116d981 --precision 'FP16' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --language en \
  --output docs/benchmarks/audio/records/A03-ASR-S1-measured.json
```

## A04: Moonshine Tiny — transcribe

Revision: `390624ed33d594443aa4aa221f5b9f283b545b5a`. Native implementation; see the linked real-checkpoint hardware evidence.

```bash
python -m benchmarks.audio.download --key A04 --output ~/mlx-one-models/moonshine-tiny
mlx-one audio transcribe --model ~/mlx-one-models/moonshine-tiny \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/moonshine-tiny \
  --checkpoint UsefulSensors/moonshine-tiny \
  --revision 390624ed33d594443aa4aa221f5b9f283b545b5a --precision 'FP32' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --output docs/benchmarks/audio/records/A04-ASR-S1-measured.json
```

## A05: Moonshine Base — transcribe

Revision: `7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf`. Native implementation; see the linked real-checkpoint hardware evidence.

```bash
python -m benchmarks.audio.download --key A05 --output ~/mlx-one-models/moonshine-base
mlx-one audio transcribe --model ~/mlx-one-models/moonshine-base \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/moonshine-base \
  --checkpoint UsefulSensors/moonshine-base \
  --revision 7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf --precision 'FP32' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --output docs/benchmarks/audio/records/A05-ASR-S1-measured.json
```

## A06: Qwen3-ASR 0.6B — transcribe

Revision: `89e96d92ba34aca20b3e29fb10cc284097d1219f`. Native implementation; see the linked real-checkpoint hardware evidence.

```bash
python -m benchmarks.audio.download --key A06 --output ~/mlx-one-models/qwen3-asr-0.6b-q8
mlx-one audio transcribe --model ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --language English --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/qwen3-asr-0.6b-q8 \
  --checkpoint mlx-community/Qwen3-ASR-0.6B-8bit \
  --revision 89e96d92ba34aca20b3e29fb10cc284097d1219f --precision 'Q8 / BF16' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --language English \
  --output docs/benchmarks/audio/records/A06-ASR-S1-measured.json
```

## A07: Qwen3-ASR 1.7B — transcribe

Revision: `a8379a2e2f9e313c9292cdf1af4055ab56d50d55`. Q8 source path implemented; historical BF16 publisher path has one punctuation anchor difference.

```bash
python -m benchmarks.audio.download --key A07 --output ~/mlx-one-models/qwen3-asr-1.7b-q8
mlx-one audio transcribe --model ~/mlx-one-models/qwen3-asr-1.7b-q8 \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --language English --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/qwen3-asr-1.7b-q8 \
  --checkpoint mlx-community/Qwen3-ASR-1.7B-8bit \
  --revision a8379a2e2f9e313c9292cdf1af4055ab56d50d55 --precision 'Q8 / BF16' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --language English \
  --output docs/benchmarks/audio/records/A07-ASR-S1-measured.json
```

## A08: Parakeet TDT v2 — transcribe

Revision: `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`. Original NeMo archive: run existing isolated converter before native inference; conversion remains WIP.

```bash
python -m benchmarks.audio.download --key A08 --output ~/mlx-one-models/parakeet-v2-archive
python benchmarks/parakeet_conversion.py \
  --archive ~/mlx-one-models/parakeet-v2-archive/parakeet-tdt-0.6b-v2.nemo \
  --reference-source /path/to/pinned-transformers \
  --output ~/mlx-one-models/parakeet-v2-native
python -m benchmarks.audio.convert_manifest --model ~/mlx-one-models/parakeet-v2-native
mlx-one audio transcribe --model ~/mlx-one-models/parakeet-v2-native \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/parakeet-v2-native \
  --checkpoint nvidia/parakeet-tdt-0.6b-v2 \
  --revision ae9ad07059c7c739ffaf932226a8fe64ae2620b0 --precision 'FP32' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --output docs/benchmarks/audio/records/A08-ASR-S1-measured.json
```

## A09: Parakeet TDT v3 — transcribe

Revision: `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`. Native implementation; see the linked real-checkpoint hardware evidence.

```bash
python -m benchmarks.audio.download --key A09 --output ~/mlx-one-models/parakeet-v3
mlx-one audio transcribe --model ~/mlx-one-models/parakeet-v3 \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/parakeet-v3 \
  --checkpoint mlx-community/parakeet-tdt-0.6b-v3 \
  --revision ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15 --precision 'FP32' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --output docs/benchmarks/audio/records/A09-ASR-S1-measured.json
```

## A10: LFM2.5 Audio 1.5B — transcribe

Revision: `55d4de65fa10fa935561aaaa995271d8b8ff2ee0`. Native source and codec tested on the pinned checkpoint; speech quality requires review.

```bash
python -m benchmarks.audio.download --key A10 --output ~/mlx-one-models/lfm2.5-audio-q4
mlx-one audio transcribe --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --checkpoint mlx-community/LFM2.5-Audio-1.5B-4bit \
  --revision 55d4de65fa10fa935561aaaa995271d8b8ff2ee0 --precision 'Q4 / BF16' \
  --task transcribe --workload ASR-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --output docs/benchmarks/audio/records/A10-ASR-S1-measured.json
```

## A11: Qwen3-ForcedAligner 0.6B — align

Revision: `0e1a68e91d815300c7c9754b2a7639378b23db15`. WIP: historical timestamp reference discrepancy unresolved; no qualification claim.

```bash
python -m benchmarks.audio.download --key A11 --output ~/mlx-one-models/qwen3-aligner-q8
mlx-one align ~/mlx-one-models/qwen3-aligner-q8 /tmp/mlx-one-audio-fixtures/librispeech-10s.wav \
  --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt --language English \
  --output alignment.json --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/qwen3-aligner-q8 \
  --checkpoint mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
  --revision 0e1a68e91d815300c7c9754b2a7639378b23db15 --precision 'Q8 / BF16' \
  --task align --workload ALIGN-S1 --fixture-language English \
  --audio /tmp/mlx-one-audio-fixtures/librispeech-10s.wav --text-file /tmp/mlx-one-audio-fixtures/librispeech.txt \
  --language English \
  --output docs/benchmarks/audio/records/A11-ALIGN-S1-measured.json
```

## A12: LFM2.5 Audio 1.5B — synthesize

Revision: `55d4de65fa10fa935561aaaa995271d8b8ff2ee0`. Native source and codec tested on the pinned checkpoint; speech quality requires review.

```bash
# Reuse the A10 download; no second copy is needed.
mlx-one audio speak --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --text "Hello from MLX-One." --voice us_male --output generated.wav --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --checkpoint mlx-community/LFM2.5-Audio-1.5B-4bit \
  --revision 55d4de65fa10fa935561aaaa995271d8b8ff2ee0 --precision 'Q4 / BF16' \
  --task synthesize --workload TTS-S1 \
  --text-file fixtures/tts-30-words.txt --voice us_male \
  --output docs/benchmarks/audio/records/A12-TTS-S1-measured.json
```

## A13: LFM2.5 Audio 1.5B — speech-to-speech

Revision: `55d4de65fa10fa935561aaaa995271d8b8ff2ee0`. Native source and codec tested on the pinned checkpoint; speech quality requires review.

```bash
# Reuse the A10 download; no second copy is needed.
mlx-one audio speak --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --audio fixtures/request-10s.wav --output response.wav --offline
python -m benchmarks.audio.run --model ~/mlx-one-models/lfm2.5-audio-q4 \
  --checkpoint mlx-community/LFM2.5-Audio-1.5B-4bit \
  --revision 55d4de65fa10fa935561aaaa995271d8b8ff2ee0 --precision 'Q4 / BF16' \
  --task speech-to-speech --workload STS-S1 \
  --audio fixtures/request-10s.wav \
  --output docs/benchmarks/audio/records/A13-STS-S1-measured.json
```
