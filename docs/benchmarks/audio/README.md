# Native audio benchmarks

Real-checkpoint testing uses a pinned Hugging Face LibriSpeech sample on an
Apple M4 with 32 GiB unified memory. Published results come from downloaded,
checksum-verified weights and native Metal inference. See the
[fixture manifest](fixtures/librispeech-s1/manifest.json),
[coverage](coverage.md), and [hardware validation report](../../validation/audio-hf-2026-10-24/report.md).
Unmeasured records retain null metrics; failed attempts are reported separately.
These short functional checks do not establish corpus accuracy or speech quality.

The [checkpoint catalog](../../../benchmarks/audio/checkpoints.json) records exact
Hub revisions, selected assets, download sizes, task support and remaining limits.
[coverage.json](coverage.json) distinguishes source support from missing performance.
[publication.json](publication.json) explicitly selects records for the README.
Full measurements and per-run outputs belong in `records/`, outside the landing page.

## Protocol

Run each invocation in a fresh process on an idle Apple Silicon Metal host with
batch size one. Record the chip, physical memory, macOS, Python and MLX versions,
checkpoint repository/full revision, hashes of every downloaded asset, and a hash
of the native implementation and runner. The manual download tool pins the revision
and creates `mlx-one-download.json`; the runner rechecks every asset hash.

Load time is measured separately, with a Metal synchronization after loading.
Use one warmup and five measured repetitions; synchronize Metal around each timed
call. Report the median of each metric from all five repetitions. Keep their raw
outputs/timings. Warmup time and model loading are excluded from processing RTF.
MLX peak memory resets before each repetition and includes resident weights. RSS
is the process lifetime high-water mark, including loading and warmup; it cannot
be interpreted as a per-request allocation or pooled with MLX memory.

Measurement types remain distinct:

- `direct-model-prepared-pcm`: one resident bundle, host PCM prepared outside timing;
  timing includes the native model-specific frontend, encoder, decoding and, for
  speech generation, waveform codec. This is a direct task pipeline measurement,
  not an isolated encoder/kernel measurement.
- `python-engine`: the public `AudioEngine`, existing inference actor and model
  pool; includes API input admission/preparation and scheduling.
- `full-file-pipeline`: one resident bundle; timed file decoding, resampling,
  model-specific frontend, native inference and output construction.

CLI/HTTP smoke timings are **not** any of these measurements. HTTP transport,
upload buffering and client playback latency need a separate named protocol.
The generator rejects incompatible hardware/software/implementation cohorts in
one table. It never pools different workloads, precisions or measurement types.

## Fixed workloads

| Workload | Input | Output / support boundary |
| --- | --- | --- |
| ASR-S1 | Fixed 10-second English clip and known transcript | Actual ASR transcript; all implemented ASR families |
| ASR-S2 | Fixed 60-second clip and known transcript | Offline ASR; resource limits may reject a family |
| ASR-S3 | Fixed 10-minute clip and known transcript | Whisper long-form only; direct/file pipeline, not the 180-second managed API |
| ASR-S4 | Fixed 10-second multilingual clip and known transcript | Separate EN/FR/DE records; supported languages only; v2/Moonshine English only |
| ASR-S5 | Fixed 10-second clip, pushed in two-second chunks | Qwen accumulated utterance only; no playback pacing or causal encoder-cache claim |
| ALIGN-S1 | Fixed 10-second clip, known text and optional independently annotated word boundaries | Qwen ForcedAligner; WIP; no ASR substitution |
| TTS-S1 | Fixed 30-word English text below | Actual generated samples at 24 kHz; one specified LFM voice |
| TTS-S2 | A pinned long text file | LFM; 512 codec frames / 40.96 seconds output bound, fail instead of truncating |
| STS-S1 | Fixed 10-second spoken input | LFM interleaved text and speech response; separate from ASR and TTS |

Use the **same exact input SHA-256** for every checkpoint within a workload. The
runner enforces input duration to within 50 ms. Commit a fixture manifest with
license/permission, source revision, file hash, language and transcript hash before
publishing. A workload name alone does not make different clips comparable.
The fixture and checkpoint download commands run only when explicitly invoked.

Fixed TTS-S1 text (30 words):

> Hello from MLX One. This short speech sample helps measure native audio generation
> on Apple Silicon. Please speak clearly and naturally while keeping every sentence
> easy to understand and follow.

RTF = processing seconds / **actual input audio seconds** for ASR/alignment. For
TTS and speech-to-speech synthesis it uses **actual generated output seconds**.
Audio sec/sec is the reciprocal. Requested durations/token budgets never enter
these denominators. Speech-to-speech input seconds are also recorded separately.
TTFA includes model input preparation, generation and waveform decode through the
first valid stable PCM chunk. First text latency remains separate in raw outputs.
Encoder/decoder/codec subtimings are retained only where the implementation emits
them; missing subtimings remain unavailable rather than being inferred from wall
clock differences. Load and peak memory include the separate LFM waveform codec.

ASR stores WER/CER using the repository's existing normalization profile. It does
not imply language detection correctness or word-timestamp quality. ASR-S5 records
first partial, final result, finalization, revised hypothesis count and retained
PCM peak bytes; the reported time is compute time with no playback delay. Alignment
records unmatched words and mean/median/p95/max absolute boundary errors only when
independent matching reference boundaries are supplied; otherwise those fields
stay null. Timestamp predictions never become invented confidence scores.

Historical audio qualification corpora in
[the existing report](../../audio-qualification-m4-32gb.md) use different fixtures,
repetition counts and reference conditions. Their aggregate RTF/accuracy results
are retained there and are not repackaged into ASR-S1 results. Text-model 128/16,
128/128 and VLM benchmarks likewise remain separate. The retired exhaustive
numerical qualification suite is not part of model admission or this workflow.

## Reproduce and publish

Follow [AUDIO_TESTING.md](../../AUDIO_TESTING.md) to download manually, smoke-test,
run a measurement and select its JSON in `publication.json`. Then run:

```bash
python -m benchmarks.audio.readme_generator --write
python -m benchmarks.audio.readme_generator --check
```

Only `MLX-ONE-AUDIO-BENCHMARK-START/END` is replaced. Text and VLM scores and the
original README design are preserved. `NOT_RUN` rows contain no fabricated
hardware, speed, latency or memory values. The generator is reusable after each
manual run; it rejects invalid records, conflicting precisions and mixed cohorts.

## Hugging Face fixture and reproduction

The primary recognition/alignment fixture is sample `1272-128104-0003`, configuration
`clean`, split `validation`, from
[hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a).
It is English read speech from LibriSpeech dev-clean, one male speaker. Attribution:
Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, *LibriSpeech: an ASR
corpus based on public domain audio books* (2015). The original recording is 9.9
seconds; exactly 0.1 seconds of trailing silence is added, without cutting speech.
Audio is mono 16 kHz PCM16. The transcript is the dataset's text, without corrections
to favor any model. WER/CER use the repository's `whisper-wer-v1` /
`whisper-cer-v1` normalization: lowercase, punctuation removal and whitespace
normalization. Proper-name spelling errors remain errors. This **single-clip** WER
is a functional check, not a LibriSpeech test-set result. [LibriSpeech is CC BY 4.0](https://www.openslr.org/12/).

The STS-S1 workload uses this same audiobook utterance as spoken input, not a
question or instruction. Its response is unconstrained and exceeds the existing
40.96-second output bound in greedy mode. A sampled diagnostic has unstable
waveform amplitude; neither is published as successful performance.

A separate **STS-HF-Q1** workload uses sample `1272-135031-0012` ("Where is my brother
now"), a complete 2.04-second question without padding. The model's response is
unconstrained, with greedy text/audio generation. This short case completes; it
remains separate from the failed 10-second workload. Both fixture hashes and the
unmodified question transcript are retained in the
[question manifest](fixtures/librispeech-s1/sts-question-manifest.json). TTS-S1 uses the fixed
30-word repository text and `us_male` voice prompt. Neither synthesis workload
has a listening/MOS score. Alignment latency can be measured without reference
word boundaries; boundary-accuracy metrics remain null.

```bash
# Optional fixture preparation dependency; not an inference dependency.
python -m pip install pyarrow
python -m benchmarks.audio.fixtures \
  --output /tmp/mlx-one-audio-fixtures \
  --cache-dir /tmp/mlx-one-audio-fixture-cache
# Download selected checkpoints explicitly; A10 is shared by A12/A13.
python -m benchmarks.audio.download --key A01 --output /tmp/mlx-one-audio-models/A01
# Download the other catalog keys to matching directories as needed.
# A08 must be converted to /tmp/mlx-one-audio-models/A08-converted first.
python -m benchmarks.audio.campaign \
  --models-root /tmp/mlx-one-audio-models \
  --fixtures /tmp/mlx-one-audio-fixtures \
  --output /tmp/mlx-one-audio-measurements --keys A01
```

The campaign is local-only, runs models serially in fresh processes, preserves
stderr/exit codes and refuses to overwrite results. Omit `--keys` to run all
13 checkpoint/tasks after downloading 11 checkpoints and converting A08.
Model loading and disk fixture decoding are outside the direct-model clock.
Language hints are `en` for Whisper, `English` for Qwen, and absent for families
without conditioning. The semantic fixture language is stored separately from
these family-specific hints. Whisper uses its native defaults (beam 5, best-of 5,
temperature fallback 0 through 1); Moonshine, Qwen and Parakeet use native greedy
decoding. LFM uses greedy text and audio codebooks with a 1,024-step synthesis
budget. Actual termination is required; budget exhaustion fails the run. The
full native-default settings and any explicit override are retained with the
source identity/record, not treated as identical decoder algorithms.

Keep model files and Hub, pip, UV and dataset caches in a directory created for
the run. After copying measurements and review artifacts, delete only those
explicitly owned downloads/caches. Never clear the user's global Hugging Face
cache or existing model directories. The hardware report records actual cleanup.

The runner can save the first measured speech waveform with `--save-first-audio`.
Saving/hashing occurs after the timed inference call. Every measured speech run
retains a SHA-256 of its actual float waveform; the saved WAV is a PCM16 review
artifact. These files and automatic Whisper Tiny round-trip diagnostics support
content review, not a MOS/intelligibility certification. The failed sampled STS
diagnostic is retained separately from performance records.
