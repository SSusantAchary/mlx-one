# Native audio: Hugging Face hardware run — 2026-10-24

All **11 primary checkpoint revisions** were downloaded into one task-owned directory and exercised on Apple M4 with 32 GiB unified memory. The 13 primary checkpoint/tasks produced 12 completed benchmark records and one failed workload. A separate short-question STS workload completed. Including separately labeled streaming, Python Engine and file-pipeline runs, **16 records / 80 measured repetitions** were retained. The README publishes 13 measured rows plus the failed primary STS row.

## Hardware, data and timing

Native MLX/Metal; MLX 0.32.2, Python 3.12.11, macOS 15.8.1. Batch one, one warmup, five measured repetitions, per-column median. Models run serially, in fresh processes, after downloads finished. Loading and direct-model fixture decoding are outside the clock; input frontend, encoder, generation and waveform codec are inside. Metal is synchronized around timing. Peak MLX resets before each repetition; RSS is process-lifetime high-water memory, not request allocation.

Data: pinned [LibriSpeech subset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a). ASR/align/primary STS use complete utterance `1272-128104-0003`, 9.9 seconds plus 0.1-second trailing silence. STS-HF-Q1 uses a different, unpadded 2.04-second question, `1272-135031-0012`. TTS uses the fixed 30-word text. The fixture preparation command reproduced the exact hashes. See the [fixture manifests](../../benchmarks/audio/fixtures/librispeech-s1/manifest.json) and [methodology](../../benchmarks/audio/README.md). LibriSpeech attribution/license are retained there.

## Primary results

| Key | Task | Status | Median processing (ms) | Audio sec/sec | Single-clip WER (%) |
| --- | --- | --- | ---: | ---: | ---: |
| [A01](../../benchmarks/audio/results/hf-2026-10-24/A01-ASR-S1.json) | transcribe | Complete; experimental | 2144.869 | 4.662 | 4.167 |
| [A02](../../benchmarks/audio/results/hf-2026-10-24/A02-ASR-S1.json) | transcribe | Complete; experimental | 7666.962 | 1.304 | 4.167 |
| [A03](../../benchmarks/audio/results/hf-2026-10-24/A03-ASR-S1.json) | transcribe | Complete; experimental | 5433.400 | 1.840 | 0.000 |
| [A04](../../benchmarks/audio/results/hf-2026-10-24/A04-ASR-S1.json) | transcribe | Complete; experimental | 117.038 | 85.443 | 4.167 |
| [A05](../../benchmarks/audio/results/hf-2026-10-24/A05-ASR-S1.json) | transcribe | Complete; experimental | 153.511 | 65.142 | 12.500 |
| [A06](../../benchmarks/audio/results/hf-2026-10-24/A06-ASR-S1.json) | transcribe | Complete; experimental | 528.726 | 18.913 | 0.000 |
| [A07](../../benchmarks/audio/results/hf-2026-10-24/A07-ASR-S1.json) | transcribe | Complete; experimental | 1017.307 | 9.830 | 0.000 |
| [A08](../../benchmarks/audio/results/hf-2026-10-24/A08-ASR-S1.json) | transcribe | Complete; experimental | 169.265 | 59.079 | 0.000 |
| [A09](../../benchmarks/audio/results/hf-2026-10-24/A09-ASR-S1.json) | transcribe | Complete; experimental | 174.019 | 57.465 | 0.000 |
| [A10](../../benchmarks/audio/results/hf-2026-10-24/A10-ASR-S1.json) | transcribe | Complete; experimental | 547.659 | 18.260 | 4.167 |
| [A11](../../benchmarks/audio/results/hf-2026-10-24/A11-ALIGN-S1.json) | align | Complete; experimental | 189.892 | 52.662 | — |
| [A12](../../benchmarks/audio/results/hf-2026-10-24/A12-TTS-S1.json) | synthesize | Complete; experimental | 3841.250 | 3.207 | — |
| [A13](../../benchmarks/audio/results/hf-2026-10-24/A13-STS-S1.json) | speech-to-speech | Failed: output bound | — | — | — |

No numerical qualification/release gates were run or reinstated. Passing a single English clip does not establish corpus WER, broader language support, timestamp accuracy, long-form support, or listening quality. Known Parakeet v2 original-NeMo parity and Qwen alignment issues remain WIP. The aligner returned no unmatched words on this fixture; boundary errors remain **null**, because independently annotated word boundaries were not supplied.

## Speech-to-speech limitation

Greedy STS-S1 exceeds the existing 512-frame / 40.96-second output bound; warmup fails explicitly. No partial generation speed, requested output length or incomplete waveform is promoted into a benchmark. A seed-zero publisher-style diagnostic (audio temperature 1, top-k 4) terminates, but raw float PCM has RMS 12.100 and peak absolute amplitude 304.140. Its metadata/log remain in `pilots/`; it is not published as successful performance. Saved PCM16 in that temporary diagnostic was clipped by serialization; raw amplitude statistics were computed before saving.

A separate 2.04-second spoken question completes with normal-amplitude audio, an unconstrained refusal response, and 15.36 seconds of generated speech. Its five-run [STS-HF-Q1 measurement](../../benchmarks/audio/results/hf-2026-10-24/A13-STS-HF-Q1-direct-model-prepared-pcm.json) is displayed separately. This supports a scoped working case and does not resolve the primary failure. TTS and short STS samples are retained beside their measurements, with verified hashes. Automatic Whisper Tiny round trips are in [final-synthesis-review.json](final-synthesis-review.json): TTS WER 6.67%, STS response WER 8.33%. These are diagnostics against intended/generated text, not human intelligibility/MOS ratings.

## Separate execution measurements

The Qwen 0.6B accumulated-utterance ASR-S5 run, Whisper Tiny Python Engine run and Whisper Tiny full-file run are retained beside the direct-model measurements with explicit `measurement_type` fields. Streaming replays five 2-second chunks without playback waits; it is not a causal encoder or an end-user network latency measurement. Different measurement types are not pooled, and their timing differences are not interpreted as transport/Engine overhead.

Real functional checks also cover repeated AudioEngine calls for Whisper/Moonshine, LFM ASR/TTS/short STS using one resident model, CLI transcription, and the actual ASGI HTTP transcription route (JSON/verbose JSON/text, unsupported SRT/VTT rejection, and authentication). All checked model pools have zero leases/resident models after close. ASGI checks are functional and do not measure socket/HTTP throughput.

## Validation and provenance

- 33 benchmark/API tests passed; 20 native integration tests passed. Two dependency deprecation warnings per suite. Ruff passes.
- Existing text and VLM publication checks still pass. README bytes outside the audio markers are unchanged.
- All 16 measured records use the same frozen implementation identity. [source-files.json](source-files.json) records source/runner hashes.
- [checkpoint-asset-manifests.json](checkpoint-asset-manifests.json) records every downloaded asset SHA-256. Parakeet v2 used the checksum-locked isolated converter with PyTorch 2.7.1 and Transformers revision `469230357aab0f2b303b0d638c1f8d06edb14184`; [conversion provenance](parakeet-v2-conversion-provenance.json) is retained. Production inference uses native MLX and no external neural runtime.
- Raw per-run transcripts, metrics, generated waveform hashes, failed attempt logs and exact commands remain outside the README. Initial unselected `NOT_RUN` placeholders in `docs/benchmarks/audio/records/` are historical planning records; current selection is explicit in `publication.json`.

## Cleanup

Verified deletion of the entire task-owned directory, including model weights, converted Parakeet assets, Hub/dataset caches, the isolated conversion environment/source and pip/UV caches: 17.18 GiB logical regular-file data. The directory is absent. See [cleanup.json](cleanup.json). Pre-existing user models, global Hugging Face caches, and the earlier source/VLM implementation directories are outside the task-owned root and are preserved.
