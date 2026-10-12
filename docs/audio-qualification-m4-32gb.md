# Native audio M4/32 GB qualification — partial results and WIP

Recorded: 2026-10-02. Host: Apple M4, 32 GiB unified memory, macOS 15.7.9
(24G830). Native execution used `~/venvs/mlx` outside the sandbox for Metal.
The complete **12-checkpoint implementation plan is not complete**. The initial
2026-10-02 results cover only the stated Distil-Whisper and original Moonshine
Tiny/Base English short-form fixture scope. Later sections retain their separate
dates and checkpoint/language limits. Stage-three work recorded on 2026-10-03 adds
native Qwen ASR/alignment implementations, but their applicable qualification
gates are not complete. Stage-four Parakeet offline implementations were added on
2026-10-04; their full qualification remains WIP. Nemotron and stages five–six
inference remain unimplemented.

## Development sequencing update — 2026-10-04

The maintainer approved continuing implementation through **Stage 4 → Stage 5 →
Stage 6** while failed or unavailable qualification gates stay **WIP**. Historical
FAIL/PASS evidence and acceptance thresholds are unchanged. Earlier statements
that Stage 3 closure was a prerequisite describe the previous sequencing policy,
not the current development gate. Qualification failures do not block another
family's implementation, but never establish checkpoint qualification.

Stage 3 WIP covers the 1.7B exact-anchor mismatch (15/16), ForcedAligner boundary
errors (up to 800 ms versus 80 ms), Qwen streaming transcript parity and missing
corpus/language gates. Experimental corrections that regress passing models
remain benchmark-only. Stage 4 now includes native Parakeet offline inference and
a frozen Nemotron contract. Stages 5–6 asset/source preparation is underway, but
they are not yet available as runtime loaders.

For Stage 6, the fixed English/Chinese intelligibility evaluator is the pinned
Qwen 1.7B **reference**, executed only in its isolated validation environment.
Unqualified native ASR is not substituted into that gate. Production synthesis
remains MLX-only; recorded fixture permission/intended-use evidence is still required.

## Checkpoint, source and fixtures

| Item | Pinned value |
| --- | --- |
| Checkpoint | [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| Revision | `8031d2e6ce6631b7fc45629dddfc00271116d981` |
| Architecture | Whisper: 32 encoder layers, 2 decoder layers, width 1280, 128 Mel bins |
| Weight layout | Publisher `model.safetensors`, float16; alternate `model.fp32.safetensors` is not merged/downloaded |
| Reference | Transformers 5.8.1, source `cc832f9055ba11c8c55f918ab4bda9472b910d48` |
| Reference environment | Isolated Python 3.12, PyTorch 2.7.1 CPU, NumPy 2.5.3, soundfile 0.13.1 |
| Native environment | Python 3.13, MLX 0.32.2, NumPy 2.2.6 |
| Dataset | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr), test-clean |
| Dataset revision | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` |
| Selection | First 100 records sorted by ID from `clean/test/0000.parquet`; first 16 are exact anchors |
| Audio | 901.1350625 seconds total; mono 16 kHz; short English utterances |
| Fixture manifest SHA-256 | `5fc089180becc98eb44d7ae45b7d92471b404d163d7ea3a68280fed2f63e0ff2` |
| Process budget | 24 GiB (`25,769,803,776` bytes) |

The native loader preserves stored float16 weights; normal float32 frontend
inputs lead to mixed/float32 intermediate computation. The CPU reference converts
the same stored weights to float32. This is not a comparison of two pure float16
implementations, nor qualification of the publisher's separate float32 export.
No quantization was used. Reference and native checkpoint runs were separate.

The [fixture manifest](../qualification/results/distil-large-v3-en-fixture-manifest.json)
records source and per-clip checksums. Audio remains in the local validation cache;
it is not added to the repository. Respect the source dataset's attribution/license.
The [Distil checkpoint contract](../qualification/results/distil-large-v3-checkpoint-contract.json)
records the selected asset hashes, raw tensor headers, configuration, preprocessing
and reference source pin. Its publisher checkpoint license is MIT, separate from
the repository's Apache-2.0 code license.

## Measured gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Strict checkpoint loading / asymmetric decoder | PASS | Pinned integration test; 32 encoder / 2 decoder layers |
| Float32 synthetic encoder | PASS | Maximum absolute error `2.384185791015625e-7` |
| Float32 synthetic decoder logits | PASS | Maximum absolute error `1.816079020500183e-8` |
| Float32 frontend | PASS | Maximum absolute error `3.24249267578125e-5`; `atol=rtol=1e-4` |
| Exact transcript anchors | PASS | 16/16 exact text matches, including case and punctuation |
| Corpus WER | PASS | Native and reference both `0.020034100596760442` (2.0034%) |
| Corpus CER | PASS | Native and reference both `0.0069051500911096195` (0.6905%) |
| Segment timestamp anchors | PASS | 16 clips; matched segments; maximum boundary difference 0 seconds |
| Process memory | PASS | Native peak resident bytes `1,731,903,488` |
| Repeated-request active allocations | PASS | All 100 calls returned to `1,512,837,140` active MLX bytes; growth 0 |

WER/CER use corpus edit counts divided by total reference units, with the existing
NFKC/lowercase/punctuation normalization. Existing `whisper-wer-v1` and
`whisper-cer-v1` macro-average evaluation profiles are unchanged. Segment end
times are clamped to original audio duration on both sides; the reference capture
also retains raw end times. Word-timestamp accuracy is **not qualified**.

Evidence: [float32 components](../qualification/results/whisper-float32-component-parity-m4.json),
[transcript comparison](../qualification/results/distil-large-v3-en-100-comparison-m4.json),
[native transcripts/resources](../qualification/results/distil-large-v3-en-100-native-m4.json),
[reference transcripts/resources](../qualification/results/distil-large-v3-en-100-reference.json),
[segment comparison](../qualification/results/distil-large-v3-timestamps-comparison-m4.json),
[native segments](../qualification/results/distil-large-v3-timestamps-native-m4.json), and
[reference segments](../qualification/results/distil-large-v3-timestamps-reference.json).

## Resource and latency scope

| Measurement | Native |
| --- | ---: |
| Checkpoint load seconds | 0.5950 |
| Total run seconds | 350.2057 |
| Per-clip processing total seconds | 346.3238 |
| Real-time factor | 0.3843 |
| Peak MLX allocation bytes | 2,789,830,800 |
| Peak process resident bytes | 1,731,903,488 |
| Active allocation growth bytes | 0 |

This measured real-time factor applies to sequential, already-recorded short
files with one loaded model, greedy decoding and disabled fallback thresholds.
It does not demonstrate a live streaming interface, cold per-file latency or
production workload throughput. Other regression activity overlapped part of the
native run. MLX peak allocation and resident memory are separate unified-memory
observations: do not add them together or infer a CPU/GPU speedup from these runs.

## Reproduction

Validation-only packages do not become production dependencies. Prepare an isolated
reference environment with PyTorch 2.7.1, Transformers 5.8.1 and soundfile 0.13.1;
fixture preparation also requires Hugging Face Hub and pyarrow. Download only the
pinned checkpoint's standard safetensors layout and its tokenizer/processor assets.
Run from the repository root, with new output paths each time:

```bash
python benchmarks/audio_fixtures.py --output /path/to/new-fixtures
python benchmarks/audio_qualification.py --backend reference \
  --manifest /path/to/new-fixtures/manifest.json --output /path/to/reference.json
# After reference exits, activate ~/venvs/mlx for the native command.
python benchmarks/audio_qualification.py --backend native \
  --manifest /path/to/new-fixtures/manifest.json --output /path/to/native.json
python benchmarks/audio_qualification.py --backend compare \
  --reference /path/to/reference.json --native /path/to/native.json \
  --output /path/to/comparison.json
```

For the separate segment gate, capture each backend with `--timestamp-capture
--limit 16` into new files, then compare with `--backend compare-timestamps`.
For component parity, run [audio_component_parity.py](../benchmarks/audio_component_parity.py)
first with `--backend reference --reference-directory NEW_DIRECTORY`, then with
`--backend native --reference-directory SAME_DIRECTORY --output NEW_RESULT`.
Failed comparison commands return a nonzero status and preserve their report.

The exact greedy checkpoint profile uses `language=en`, 128 generated-token
limit, timestamp generation, no previous-text conditioning, and no fallback/
no-speech thresholds. The historical Whisper default decode profile is unchanged.

## Moonshine Tiny/Base stage-two results

Both original checkpoints now have native raw-waveform encoders, autoregressive
cached decoders, strict safetensors/tokenizer loading and Python/CLI offline
transcription. The reference uses the same isolated Transformers 5.8.1 source
commit recorded above. Each run uses the same 100 English fixture clips and 16
anchors, with greedy decoding, BOS 1, EOS/pad 2 and 193 generated-token budget.
No audio normalization, Mel extraction, quantization or reduced precision is used.
The publisher float32 weights are retained by both implementations.

| Checkpoint | Revision | Native architecture |
| --- | --- | --- |
| [Tiny](https://huggingface.co/moonshine-ai/moonshine-tiny) | `390624ed33d594443aa4aa221f5b9f283b545b5a` | Width 288; 6 encoder / 6 decoder layers; 8 heads; partial rotary fraction 0.9 |
| [Base](https://huggingface.co/moonshine-ai/moonshine-base) | `7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf` | Width 416; 8 encoder / 8 decoder layers; 8 heads; partial rotary fraction 0.62 |

Each uses valid 127/7/3 waveform convolutions at strides 64/3/2, one-group
GroupNorm, a GELU encoder MLP and a SiLU-gated decoder MLP. Configuration,
preprocessing, raw tensor names/shapes/dtypes and SHA-256 hashes of weights and
tokenizer assets are recorded in the [Tiny checkpoint contract](../qualification/results/moonshine-tiny-checkpoint-contract.json)
and [Base checkpoint contract](../qualification/results/moonshine-base-checkpoint-contract.json).
Both publisher model cards identify the checkpoint license as MIT; adapted
Transformers code notices and mlx-one's Apache-2.0 license remain separate.

| Gate / measurement | Tiny | Base |
| --- | ---: | ---: |
| Exact greedy anchor text | 16/16 PASS | 16/16 PASS |
| Native / reference WER | 0.032821824381926684 / same | 0.028132992327365727 / same |
| Native / reference CER | 0.013714395319842716 / same | 0.010165915411911385 / same |
| Float32 waveform frontend max absolute error | `1.9371509552001953e-7` | `3.725290298461914e-7` |
| Float32 encoder max absolute error | `2.86102294921875e-6` | `4.023313522338867e-6` |
| Float32 full decoder logits max absolute error | `2.2351741790771484e-7` | `3.3155083656311035e-7` |
| Float32 cached decoder logits max absolute error | `2.2351741790771484e-7` | `3.520399332046509e-7` |
| Checkpoint load seconds | 0.3445 | 0.3960 |
| Per-clip processing total seconds | 23.7140 | 29.7643 |
| Measured file-transcription real-time factor | 0.02632 | 0.03303 |
| Peak MLX allocation bytes | 421,650,712 | 756,419,136 |
| Peak process resident bytes | 298,516,480 | 442,253,312 |
| Active allocation growth over 100 requests | 0 | 0 |

The synthetic tests exercise original head widths/rotary dimensions, waveform
frontend, masked batch attention and cached decoding at `atol=rtol=1e-4`.
Checkpoint transcripts were evaluated independently from those components.
Exact agreement is required for the fixed anchors; equal normalized WER/CER is
not a claim of verbatim agreement on every non-anchor clip.

Evidence for Tiny: [component gates](../qualification/results/moonshine-tiny-component-comparison.json),
[checkpoint gates](../qualification/results/moonshine-tiny-comparison.json),
[native run](../qualification/results/moonshine-tiny-native.json) and
[reference run](../qualification/results/moonshine-tiny-reference.json).
Evidence for Base: [component gates](../qualification/results/moonshine-base-component-comparison.json),
[checkpoint gates](../qualification/results/moonshine-base-comparison.json),
[native run](../qualification/results/moonshine-base-native.json) and
[reference run](../qualification/results/moonshine-base-reference.json).

Native active allocations return to the loaded-model baseline after every
request, but the shared MLX allocator retains reusable cached buffers. Its
cached-byte values grow during these varying-length workloads; zero active
growth does not mean zero allocator retention. Requests do not clear shared
allocator state. Resident and MLX metrics are distinct observations, not additive.

The measured factors are below one for this already-recorded short English
workload only; they do not qualify live streaming or long recordings. No
language detection, timestamps, alignment, diarization or Moonshine streaming
interface is implemented. Empty input and unsupported/short/nonfinite input
behavior are covered by network-free and native tests, not treated as speech
quality fixtures. The 24 GiB process gate and conservative dense-encoder
preflight reject resource limits explicitly without dropping audio.

### Moonshine reproduction

Use [the family guide](../src/mlx_one/models/audio/moonshine/README.md) for
installation and normal Python/CLI workflows. Reuse the fixture manifest above.
Run each reference process to completion before launching the native process:

```bash
# Isolated CPU reference environment:
python benchmarks/audio_qualification.py --backend reference \
  --model moonshine-ai/moonshine-tiny \
  --revision 390624ed33d594443aa4aa221f5b9f283b545b5a \
  --manifest /path/to/fixtures/manifest.json --output /path/to/new-reference.json
# Native ~/venvs/mlx environment after reference exits:
python benchmarks/audio_qualification.py --backend native \
  --model moonshine-ai/moonshine-tiny \
  --revision 390624ed33d594443aa4aa221f5b9f283b545b5a \
  --manifest /path/to/fixtures/manifest.json --output /path/to/new-native.json
python benchmarks/audio_qualification.py --backend compare \
  --reference /path/to/new-reference.json --native /path/to/new-native.json \
  --output /path/to/new-comparison.json
```

Repeat independently with Base and its pinned revision. For components, run
[moonshine_component_parity.py](../benchmarks/moonshine_component_parity.py) with
`--variant tiny` or `--variant base`: reference exports into a new directory,
then native consumes it. [audio_checkpoint_contract.py](../benchmarks/audio_checkpoint_contract.py)
records safe tensor headers and selected asset hashes without backend execution.
All harness outputs refuse existing targets. No audio or model weights are
checked into the repository.

### Regression execution

After family dispatch changes, 21 opt-in Metal tests passed (17 Whisper and 4
Moonshine), including the pinned Distil checkpoint tests, cache isolation and
cancellation/limit cleanup. Real offline JSON CLI transcription also ran for
Distil-Whisper and Moonshine Tiny. The final non-GPU suite passed 449 tests with
20 skipped; targeted Ruff, Python compilation and `git diff --check` passed.
Local documentation links resolve, all 24 registered types have a linked guide
(21 guide files; SAM3 shares one), and 32 documented Python blocks parse.
HTTP transcription remains Whisper-only, with its existing service behavior
and result schema. The CLI's pre-existing late segmentation import still triggers
Ruff E402 when linting the whole CLI file; unrelated CLI layout is preserved.

## Qwen stage-three implementation and qualification in progress

Added native MLX implementations for both ASR sizes and the distinct known-text
ForcedAligner. Production never imports the upstream reference package, PyTorch,
Transformers or vLLM. Backend-free configurations/registry imports, strict tensor
and shard contracts, checkpoint tokenizer/chat-template handling, typed results,
Python/CLI integration and bounded accumulated-utterance sessions are implemented.
These implementation statements do **not** establish checkpoint qualification.

| Checkpoint | Frozen revision | Contract |
| --- | --- | --- |
| Qwen3-ASR 0.6B | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | [Assets/tensors](../qualification/results/qwen3-asr-0.6b-checkpoint-contract.json) |
| Qwen3-ASR 1.7B | `7278e1e70fe206f11671096ffdd38061171dd6e5` | [Assets/tensors](../qualification/results/qwen3-asr-1.7b-checkpoint-contract.json) |
| Qwen3-ForcedAligner 0.6B | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | [Assets/tensors](../qualification/results/qwen3-forced-aligner-checkpoint-contract.json) |

The separate reference environment uses Python 3.12, PyTorch 2.7.1 and
Transformers 4.57.6 with Qwen source commit
`7c6daf77a2421100f5fb066495372c00129d39ff`. It does not upgrade the historical
Whisper/Moonshine environment. Both checkpoint implementations retain the
publisher's BF16 weights. Synthetic component tests use float32 separately.
This reference scope is **SDPA/global audio attention**, not the upstream
FlashAttention/windowed path. The source and publisher checkpoints are Apache-2.0.

The [initial 16-clip diagnostic](../qualification/results/qwen3-asr-0.6b-en-16-initial-diagnostic.json)
failed: 15/16 exact Qwen 0.6B anchors matched. Clip `1089-134686-0012` differed
in sentence punctuation/capitalization (period versus semicolon), despite equal
normalized WER/CER. The exact gate is not normalized or weakened. This diagnostic
also lacks the required full corpus and is explicitly **not qualification**.
Subsequent rounding investigations and full-corpus runs preserve new output
artifacts rather than overwrite this failure. Initial
[float32 synthetic component evidence](../qualification/results/qwen3-audio-float32-component-parity-m4.json)
passed independently; it cannot replace checkpoint or streaming gates.

### Qwen3-ASR 0.6B English offline results

The latest implementation preserves the pinned reference's operation order for
Hann-window coefficients, sinusoidal-position frequencies and rotary inverse
frequencies. The resulting BF16 checkpoint run now passes all **English offline**
gates, including the previously failing punctuation anchor. The earlier failure
remains recorded; other checkpoints, languages and streaming are not qualified
by this result.

| Gate / measurement | Latest native result |
| --- | ---: |
| English clips / exact anchors | 100 / 16 of 16 PASS |
| Native / reference corpus WER | 0.014066496163682864 / 0.014492753623188406 |
| Native / reference corpus CER | 0.004603433394073079 / 0.004891147981202647 |
| Float32 frontend maximum absolute difference | `7.152557373046875e-5` |
| Float32 component and tokenizer gates | PASS at `atol=rtol=1e-4`; three tokenizer fixtures agree exactly |
| Load seconds | 0.7326 |
| Run wall seconds | 134.1515 |
| Measured file-transcription real-time factor | 0.14322 |
| Peak resident bytes / 24 GiB process gate | 2,149,138,432 / PASS |
| Peak MLX active allocation bytes | 2,238,074,152 |
| Active bytes after every request | 1,564,880,940 |
| Active allocation growth across 100 requests | 0 |
| Final shared allocator cached bytes | 12,138,387,928 |

Evidence: [latest scoped checkpoint gates](../qualification/results/qwen3-asr-0.6b-en-100-comparison-m4.json),
[native corpus/resources](../qualification/results/qwen3-asr-0.6b-en-100-native-m4.json),
[reference corpus/resources](../qualification/results/qwen3-asr-0.6b-en-100-reference.json),
[latest float32 component gates](../qualification/results/qwen3-audio-float32-component-parity-m4-v3.json),
and [component fixture/source checksums](../qualification/results/qwen3-audio-float32-component-reference-v3.json).
The existing 100 English fixtures and 16 anchors are unchanged. WER/CER are
normalized corpus metrics; exact case/punctuation is separately enforced on
anchors. Non-anchor predictions need not be verbatim identical.

The sub-one real-time factor applies only to sequential, already-recorded short
English utterances, not live streaming or production throughput. Shared allocator
cached buffers grow during varying-length requests; zero active growth is not
zero allocator retention. Cleanup never clears shared allocator resources.
Resident, active and cached metrics are distinct observations, not additive.

Multilingual fixture selection is frozen before inference: FLEURS revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, first 25 test recordings sorted by
sentence ID and source-row index in each language, without deduplicating repeated
sentence IDs. The [French](../qualification/results/qwen3-fleurs-fr-fr-25-fixture-manifest.json),
[German](../qualification/results/qwen3-fleurs-de-de-25-fixture-manifest.json), and
[Mandarin](../qualification/results/qwen3-fleurs-cmn-hans-cn-25-fixture-manifest.json)
manifests preserve audio hashes, source shard hash, row selection and CC-BY-4.0
attribution metadata. Mandarin is additional scope for ASR 1.7B. Prepared manifests
alone are **not evidence of passing model runs**. Repository JSON copies are
human-readable representations; each run records the byte hash of its actual
input manifest. Audio remains outside the repository.

### Additional stage-three checkpoint runs and failures

| Checkpoint / scope | Recorded outcome | Evidence |
| --- | --- | --- |
| ASR 0.6B, French offline, 25 FLEURS clips | PASS: WER native/reference `0.09653092006033183` / `0.09803921568627451`; CER `0.02762119503945885` / `0.02846674182638106` | [Gates](../qualification/results/qwen3-asr-0.6b-fr-25-comparison-m4.json), [native](../qualification/results/qwen3-asr-0.6b-fr-25-native-m4.json), [reference](../qualification/results/qwen3-asr-0.6b-fr-25-reference.json) |
| ASR 1.7B, 16 English anchors only | FAIL: 15/16 exact anchors; punctuation mismatch at `1089-134686-0001`; required 100-clip corpus absent | [Diagnostic gates](../qualification/results/qwen3-asr-1.7b-en-16-comparison-diagnostic-m4.json), [native](../qualification/results/qwen3-asr-1.7b-en-16-native-diagnostic-m4.json), [reference](../qualification/results/qwen3-asr-1.7b-en-16-reference-diagnostic.json) |
| ForcedAligner, 100 English clips | FAIL: text units/order agree, but eight boundaries across six clips exceed one 80-ms tick; maximum difference 800 ms | [Detailed failed gates](../qualification/results/qwen3-forced-aligner-en-100-comparison-m4-v2.json), [native](../qualification/results/qwen3-forced-aligner-en-100-native-m4.json), [reference](../qualification/results/qwen3-forced-aligner-en-100-reference.json) |
| ASR 0.6B, two English streaming utterances | FAIL: complete hypotheses and one final anchor differ in colon/comma punctuation; required full corpus absent | [Diagnostic gates](../qualification/results/qwen3-asr-0.6b-en-2-stream-comparison-diagnostic-m4.json), [native](../qualification/results/qwen3-asr-0.6b-en-2-stream-native-diagnostic-m4.json), [reference](../qualification/results/qwen3-asr-0.6b-en-2-stream-reference-diagnostic.json) |

French has no additional exact anchors; the fixed English anchors enforce that
checkpoint's separate exact-text gate. These language/mode scopes are not merged
into a claim of complete Qwen qualification. German and Mandarin model runs,
the full 1.7B corpus, full streaming corpora and the remaining alignment languages
are still pending. No checkpoint was substituted and no thresholds were weakened.

All three native checkpoint runs fit the 24-GiB resident-memory gate and showed
zero active-allocation growth across requests. Peak resident bytes were
`2,123,005,952` for French 0.6B, `4,924,227,584` for the 1.7B diagnostic, and
`2,071,658,496` for alignment. These resource passes do not override quality failures.

The [alignment head diagnostic](../qualification/results/qwen3-forced-aligner-selected-full-head-diagnostic-m4.json)
found identical native logits for selected-position and full-sequence projections
on all six affected clips. Removing the projection optimization does not resolve
the timestamp failure. The residual reference/native classification difference
remains unresolved; no confidence or alternative timestamp is invented to hide it.

Qwen sessions reprocess accumulated audio; they do not retain causal encoder
caches. The implementation preserves two-second chunks, two initially unfixed
chunks, five-token rollback, the distinct finish-time rollback, and the explicit
30-second utterance limit. Events contain complete current hypotheses, not text
deltas. The validation helper applies the pinned upstream rollback/buffer
algorithm around its Torch SDPA decoder: it does **not** claim vLLM-kernel parity.
The [native lifecycle diagnostic](../qualification/results/qwen3-asr-0.6b-stream-lifecycle-diagnostic-m4.json)
passes on two real English utterances: 17-/16,000-/32,000-sample partitions produce
identical complete hypotheses, repeated reset/finish works, one session survives
another's cancellation, three mid-decode cancellations release request state, and
over-limit input is rejected without mutating state. All cleanup observations are
`1,564,880,940` active bytes, 36 bytes above the fresh-load baseline; allocator
cached buffers remain shared and are not cleared. This is lifecycle/resource
evidence only, not passing reference streaming quality or a full corpus.

Use [the Qwen guide](../src/mlx_one/models/audio/qwen3_asr/README.md) for the
implemented API/CLI and asset requirements. The
[Qwen checkpoint harness](../benchmarks/qwen_audio_qualification.py) records
one checkpoint/task/language/mode at a time and exits nonzero for failed gates.
Reference output must be persisted before native execution. At the time of this
run, Stage 3 closure was required before Stage 4; the sequencing update above
supersedes that requirement without changing these historical results.

The latest non-GPU suite passes **479 tests, 21 skipped**; 24 selected opt-in
Metal architecture/regression tests pass (Qwen, original Moonshine and Whisper,
including the already-cached pinned Whisper checkpoints).
These tests are distinct from checkpoint quality gates. The documentation checker
covers all 26 registry types with 22 linked guides, 35 Python examples, local-file
links, audio imports/signatures and audio CLI options. Targeted Ruff and
`git diff --check` pass; the historical CLI E402 caveat above remains unchanged.

## Stage-three closure investigation — rejected numerical candidate

The 2026-10-03 closure investigation added a
[checkpoint layer probe](../benchmarks/qwen_audio_layer_probe.py) and an explicit
[validation-only rounding candidate](../benchmarks/qwen_audio_rounding_candidate.py).
Neither changes production loading or default inference. The candidate was
tested, failed unchanged gates, and was **not promoted into production**.

One previously failing English aligner clip (`1089-134686-0005`) was captured in
the pinned reference environment before native Metal diagnostics. The
[reference manifest](../qualification/results/qwen3-forced-aligner-bf16-layer-reference.json)
records the checkpoint/source/fixture checksums; the large activation NPZ remains
outside the repository. The
[native layer diagnostic](../qualification/results/qwen3-forced-aligner-bf16-layer-candidate-m4.json)
isolates operators using identical reference inputs. Convolution bias and
LayerNorm affine rounding account for substantial differences. Reconstructing
the pinned CPU BF16 attention sequence reduces first-layer attention differences
from 71,557 values to six, but does not establish whole-checkpoint parity.
The reference sequence is specific to
[PyTorch 2.7.1 CPU flash attention](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp);
it must not be advertised as GPU-reference or vLLM parity.

| Candidate checkpoint / scope | Outcome | Evidence |
| --- | --- | --- |
| Qwen ASR 0.6B, English 100 | FAIL: 15/16 exact anchors; WER/CER, budget and cleanup pass | [Comparison](../qualification/results/qwen3-asr-0.6b-en-100-comparison-m4-v2.json) |
| Qwen ASR 0.6B, French 25 | PASS for this language/offline candidate scope only | [Comparison](../qualification/results/qwen3-asr-0.6b-fr-25-comparison-m4-v2.json) |
| Qwen ASR 1.7B, English 16 diagnostic | FAIL: 15/16 exact anchors; required full corpus absent | [Comparison](../qualification/results/qwen3-asr-1.7b-en-16-comparison-diagnostic-m4-v3.json) |
| ForcedAligner, English 100 | FAIL: six boundaries across five clips; maximum 480 ms exceeds 80-ms gate | [Comparison](../qualification/results/qwen3-forced-aligner-en-100-comparison-m4-v4.json) |

Intermediate convolution/normalization-only experiments are separately labelled
`audio-fp32-affine-convolution`; the complete experimental sequence is labelled
`cpu-bf16-rounding`. New harness captures record native implementation hashes.
Reference runs reject the native-only candidate flag. To reproduce an experiment,
add `--numerical-candidate` to a **native qualification harness** invocation,
never to the production CLI or a reference run. Comparison output retains that
label so failed-candidate results cannot be mistaken for production evidence.

The established production implementation and historical English/French passes
remain intact. A fresh [default English 100 comparison](../qualification/results/qwen3-asr-0.6b-en-100-comparison-m4-v3.json)
passes 16/16 anchors and WER/CER/resource gates after isolating the candidate.
The [native run](../qualification/results/qwen3-asr-0.6b-en-100-native-m4-v3.json)
records a 2,170,142,720-byte peak resident size, 2,238,074,152-byte MLX peak,
zero active-allocation cleanup growth and the native implementation hashes.
The [fresh float32 component comparison](../qualification/results/qwen3-audio-float32-component-parity-m4-v4.json)
also passes. The latest targeted suites pass 160 network-free tests and 27 Metal
tests, including explicit candidate/model isolation and existing Whisper/Moonshine
regressions. The full default regression suite passes **485 tests, 21 skipped**;
documentation links/examples, targeted Ruff and `git diff --check` also pass.
These do not establish checkpoint or streaming qualification.
Stage three still requires the outstanding alignment/transcript/streaming parity
and corpus/language gates; stages four–six remain unimplemented.

## Stage-three follow-up — decoder ties and isolated timestamp classifier

The follow-up investigation keeps the existing production model and acceptance
thresholds unchanged. The [token probe](../benchmarks/qwen_audio_probe.py) now
supports both ASR sizes, a teacher-forced prefix, frozen checkpoint/source
validation, and explicit score margins/ties. Reference-audio injection is an
explicit **diagnostic intervention only**: it is never available through the
runtime or qualification harness and cannot establish checkpoint qualification.

For the failing 1.7B anchor `1089-134686-0001`, after `Stuff it into you`, the
pinned CPU reference gives token 13 (`.`) and token 1335 (`,"`) the same cached
BF16 score, `40.0`. The reference chooses the lower token ID. Experiments with
split-scaling text SDPA, compensated linear accumulation, audio LayerNorm, and
the earlier audio-rounding sequence did not eliminate the cached token mismatch.
Supplying the saved reference audio encoding in a diagnostic still leaves the
decoder mismatch; preprocessing is therefore not its only source. These
observations do not authorize special-casing this clip, adjusting greedy scores,
removing punctuation, replacing the oracle, or weakening the exact-anchor gate.
The [default native token diagnostic](../qualification/results/qwen3-asr-1.7b-token-native-diagnostic-m4.json)
records a native cached score of `39.75` for token 13 versus `40.25` for token
1335, the frozen input-fixture hash, teacher-prefix IDs, and source hashes.
This is not a tie-breaking bug: the native score ordering itself differs.

The [text math candidate](../benchmarks/qwen_audio_text_candidate.py) mirrors the
split query/key scaling of the pinned CPU GQA math kernel. On identical saved
aligner attention inputs, it reduces differing first-layer BF16 attention values
from 173 to 16, but the 1.7B 16-anchor and 0.6B two-utterance streaming diagnostics
still fail exact text. It is validation-only and is not promoted to production.
Compensated linear accumulation and LayerNorm-only experiments likewise remain
diagnostics, not qualified runtime alternatives.

The narrower LayerNorm-only correction was also tested against the entire
passing 0.6B English corpus before considering promotion. Its
[100-clip candidate comparison](../qualification/results/qwen3-asr-0.6b-layernorm-en-100-comparison-candidate-m4.json)
fails the unchanged exact-anchor gate: **15/16**, with mismatch
`1089-134686-0012`. Aggregate WER/CER, the 24-GiB process budget, and cleanup pass.
The [native capture](../qualification/results/qwen3-asr-0.6b-layernorm-en-100-native-candidate-m4.json)
is explicitly labelled `audio-fp32-layernorm`; the production loader/model remain
unchanged, and the passing default 0.6B result is preserved. Reduced operator
differences or passing aggregate accuracy do not override the failed anchor.

The [isolated timestamp classifier probe](../benchmarks/qwen_alignment_classifier_probe.py)
uses frozen reference hidden states, preserving BF16 weights and output dtype.
It loads only the classifier tensor, not another copy of the full checkpoint.
The [reference capture](../qualification/results/qwen3-forced-aligner-classifier-reference-diagnostic.json)
and [Metal comparison](../qualification/results/qwen3-forced-aligner-classifier-native-diagnostic-m4.json)
agree on all **44 timestamp classes** for `1089-134686-0005`. Fifteen individual
logits differ, with a maximum absolute difference of `0.0625`; this is a BF16
diagnostic, not a passing float32 component gate or checkpoint qualification.
The comparison links the repository metadata hash and separately retains the
original temporary capture hash. Weight/activation NPZ files stay outside the
repository.

Compared with the historical end-to-end native class vector, exactly one raw
timestamp class differs on this clip: position 29 is 77 in the reference versus
70 natively. Applying the same timestamp repair produces `UPON` spans of
`[5.28, 5.28]` and `[5.28, 5.60]` seconds respectively. This localizes the failure
upstream of the classifier on this clip and explains why a classification change
can alter the repaired boundary. It does not resolve the other failing clips or
justify changing timestamp repair to hide the error.

### Completed 1.7B English corpus — still fails exact anchors

The 2026-10-03–04 run completes the previously missing **100 English clips**.
The [reference corpus](../qualification/results/qwen3-asr-1.7b-en-100-reference.json)
was persisted before the [native Metal corpus](../qualification/results/qwen3-asr-1.7b-en-100-native-m4.json).
The [full comparison](../qualification/results/qwen3-asr-1.7b-en-100-comparison-m4.json)
is **FAIL**: the sole failed gate is the unchanged exact-anchor requirement,
with mismatch `1089-134686-0001` and **15/16** matches. Completing the corpus does
not turn this failure into qualification.

| Metric | Native M4 | Pinned CPU reference |
| --- | ---: | ---: |
| English clips | 100 | 100 |
| Corpus WER | `0.011935208866155157` | `0.011935208866155157` |
| Corpus CER | `0.003356670183178287` | `0.003356670183178287` |
| Total run seconds | `200.26269100001082` | `2170.4585750410333` |
| Peak resident bytes | `4,968,841,216` | `6,124,879,872` |
| Peak MLX active allocation bytes | `4,763,597,608` | n/a |
| Active-allocation cleanup growth bytes | `0` | n/a |

Native load time is `1.9805277910199948` seconds. Measured file-transcription RTF
is `0.21551231792403308`; it is not a live-streaming or first-partial-latency
claim. Both process peaks meet the 24-GiB budget. Shared allocator cached buffers
remain separate from active allocation and are not cleared between requests.
The native model and preprocessing hashes remain the established production
hashes recorded above; no experimental operator was enabled for this run.

The current regressions pass **496 network-free tests, 21 skipped**, and **30
opt-in Metal tests**, including existing Whisper/Moonshine checkpoint regressions.
Documentation links/examples, targeted Ruff, and `git diff --check` pass. These
are not substitutes for the failed quality gates.

Remaining multilingual/streaming gates are still separate: the 1.7B French,
German, Mandarin and streaming corpora, 0.6B German/full streaming corpora, and
remaining alignment languages are not newly qualified. The aligner's historical
80-ms gate and streaming exact-text diagnostic still fail and remain **WIP**.
The later sequencing update permits Stage 4–6 development despite these failures.

## Remaining implementation matrix

| Stage | Checkpoints | Current status |
| --- | --- | --- |
| 1 | Distil-Whisper Large v3 | Recorded English short-form gates above pass; broader/word-timestamp qualification pending |
| 2 | Moonshine Tiny / Base | Recorded float32 components and M4 English short-form gates pass; streaming/long-form unqualified |
| 3 | Qwen3-ASR 0.6B / 1.7B; Qwen3-ForcedAligner 0.6B | Implemented; **WIP**: 0.6B English/French offline gates pass; 1.7B anchors, alignment timestamps and streaming diagnostic fail; remaining gates pending |
| 4 | Parakeet TDT v2 / v3; Nemotron 3.5 ASR 0.6B | Parakeet offline implemented, scoped v2 gates pass; **WIP** overall. Nemotron contract frozen; native streaming unimplemented |
| 5 | Voxtral Mini 3B; Voxtral Realtime 4B | Tensor/Tekken/processor contracts frozen; **WIP / not implemented** native transcription and cached streaming |
| 6 | Original VibeVoice 1.5B multi-speaker TTS | Tensor contract, original source/tokenizer pinned; **WIP / not implemented** native synthesis, parity and publisher-fixture permission/intended-use evidence |

## Parakeet Stage Four progress — 2026-10-04

Native float32 offline Parakeet TDT v2/v3 loading, FastConformer, recurrent
predictor, joint token/duration heads, greedy decoding and token timestamp
records are implemented. Streaming and word timestamps are explicitly rejected.
The [guide](../src/mlx_one/models/audio/parakeet/README.md) documents the supported
asset layout and v2's isolated conversion. This is **WIP** overall, not closure
of Stage 4 or of the full audio plan.

### Frozen contracts and component gate

- [v2 contract](../qualification/results/parakeet-tdt-v2-checkpoint-contract.json):
  `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`; original archive SHA-256
  `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`.
- [v3 contract](../qualification/results/parakeet-tdt-v3-checkpoint-contract.json):
  `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`; canonical `model.safetensors`.
- Reference Transformers commit `469230357aab0f2b303b0d638c1f8d06edb14184`,
  PyTorch 2.7.1 CPU in a new isolated Stage 4 environment. Earlier environments
  were not upgraded. The persisted
  [reference component manifest](../qualification/results/parakeet-float32-component-reference.json)
  includes dependency versions and the fixture checksum.
- [Initial component result](../qualification/results/parakeet-float32-component-parity-m4.json)
  failed silent normalization by 0.158806. A centered FP32 mean reduction fixes
  summation residuals without changing the normalization formula or tolerance.
  [Current component result](../qualification/results/parakeet-float32-component-parity-m4-v2.json)
  passes all tested components at `atol=rtol=1e-4`; random frontend maximum absolute
  difference is 0.000031477, silence and Mel filter differences are zero.

### v2 English short-form comparison

The same checksummed 100 English LibriSpeech clips (901.1350625 seconds) and 16
anchors used above were run sequentially in the frozen converted-reference and
native Metal environments. v2's lexical IDs/blank behavior are preserved rather
than importing v3's vocabulary. The comparison is against the **converted
Transformers reference**, not a recorded original NeMo inference run.

| Metric | Recorded result |
| --- | --- |
| Exact anchors | 16/16 PASS |
| Reference / native WER | 1.7476556% / 1.7476556% |
| Reference / native CER | 0.4987053% / 0.4987053% |
| Load time | 2.311 seconds |
| Native processing time / audio duration | 36.182 / 901.135 seconds |
| Offline file real-time factor | 0.04015; no streaming-performance claim |
| Peak MLX allocation | 2,660,926,980 bytes |
| Peak process memory | 2,660,958,208 bytes; below 24 GiB |
| Active MLX growth after repeated calls | 0 bytes; shared allocator cache is not cleared |

Evidence: [reference run](../qualification/results/parakeet-tdt-v2-en-100-reference.json),
[native run](../qualification/results/parakeet-tdt-v2-en-100-native-m4.json),
[comparison](../qualification/results/parakeet-tdt-v2-en-100-comparison-m4.json).
Original NeMo frontend/decoding parity, token/word timestamp accuracy and broader
workloads remain **WIP**. These scoped passing gates do not qualify the whole model.

### v3 English, French and German short-form comparisons

The independent v3 English 100-clip gate passes, including **16/16 exact
anchors**, matching reference/native WER 1.9607843% and CER 0.4507529%.
Load time was 0.769 seconds, native processing 39.626 seconds for 901.135
seconds of audio (offline RTF 0.04397), peak MLX allocation 2,697,672,228 bytes,
peak process memory 2,702,147,584 bytes and active allocation growth zero.

Evidence: [reference](../qualification/results/parakeet-tdt-v3-en-100-reference.json),
[native](../qualification/results/parakeet-tdt-v3-en-100-native-m4.json),
[comparison](../qualification/results/parakeet-tdt-v3-en-100-comparison-m4.json).

The existing checksummed 25-clip French and German selections from FLEURS
`70bb2e84b976b7e960aa89f1c648e09c59f894dd` were captured in the same pinned
reference before native execution. No explicit language conditioning was passed
to native v3. Only these two non-English subsets are covered; this does not
qualify every publisher language or long-form audio.

| Subset | Clips | Reference / native WER | Reference / native CER | Native offline RTF | Active MLX growth |
| --- | ---: | --- | --- | ---: | ---: |
| French | 25 | 6.9381599% / 6.9381599% | 1.8038331% / 1.8038331% | 0.04094 | 0 bytes |
| German | 25 | 5.9590317% / 5.9590317% | 1.3465854% / 1.3465854% | 0.03567 | 0 bytes |

French evidence: [reference](../qualification/results/parakeet-tdt-v3-fr-25-reference.json),
[native](../qualification/results/parakeet-tdt-v3-fr-25-native-m4.json),
[comparison](../qualification/results/parakeet-tdt-v3-fr-25-comparison-m4.json).
German evidence: [reference](../qualification/results/parakeet-tdt-v3-de-25-reference.json),
[native](../qualification/results/parakeet-tdt-v3-de-25-native-m4.json),
[comparison](../qualification/results/parakeet-tdt-v3-de-25-comparison-m4.json).
The fixture hashes and full reference/native dependency versions are retained in
each run. Peak process memory remained below 2.71 GB in all three subsets.
Token/word timestamp qualification and broader workloads remain **WIP**.

### Nemotron and later stages

The [Nemotron contract](../qualification/results/nemotron-3.5-asr-checkpoint-contract.json)
freezes 655 canonical tensors, tokenizer/processor and publisher safety/bias/privacy
assets at `ea30d66debe3740a08b573244286791d423d6b3e`, with official source commit
`469230357aab0f2b303b0d638c1f8d06edb14184`. The pinned model card specifies
OpenMDW-1.1, separate from mlx-one's license. Cache-aware frontend, convolution,
attention and RNNT state, language prompts, three-token lookahead and all native
checkpoint/streaming gates are **WIP / not implemented**; no runtime task is advertised.
The [initial 16-anchor reference capture](../qualification/results/nemotron-3.5-asr-en-16-reference.json)
was persisted in a **separate** Python 3.12 environment at the frozen source,
using explicit `en-US` prompts, three-token lookahead and float32 CPU inference.
Its complete dependency versions are recorded. Load time was 4.854 seconds and
peak process memory 3,872,669,696 bytes. These are reference measurements, not
native performance or a passing qualification gate. Automatic detection,
full corpora, non-English subsets and streaming reference captures remain WIP.

All approved Stage 4–6 authoritative exports have now been downloaded to the
temporary qualification cache and frozen without loading their neural backends:

| Checkpoint | Authoritative export | Frozen tensor dtype/count | Contract |
| --- | --- | --- | --- |
| Parakeet v2 | Isolated converted canonical safetensors | 699 F32 + 24 I64 | [v2](../qualification/results/parakeet-tdt-v2-checkpoint-contract.json) |
| Parakeet v3 | `model.safetensors` | 699 F32 + 24 I64 | [v3](../qualification/results/parakeet-tdt-v3-checkpoint-contract.json) |
| Nemotron 3.5 ASR | `model.safetensors` | 655 F32 | [Nemotron](../qualification/results/nemotron-3.5-asr-checkpoint-contract.json) |
| Voxtral Mini 3B | Indexed two HF shards; no consolidated export | 761 BF16 + 1 F32 | [Mini](../qualification/results/voxtral-mini-3b-checkpoint-contract.json) |
| Voxtral Realtime 4B | `model.safetensors`; no consolidated export | 711 BF16 | [Realtime](../qualification/results/voxtral-realtime-4b-checkpoint-contract.json) |
| Original VibeVoice 1.5B | Indexed three HF shards | 1,204 BF16 | [Original TTS](../qualification/results/vibevoice-1.5b-checkpoint-contract.json) |

The contracts record hashes, raw names/shapes/dtypes, shard assignments,
configuration, family-specific processor/tokenizer assets and reference-source
pins. Voxtral includes separately hashed `tekken.json` and `params.json`.
VibeVoice includes separately pinned Qwen tokenizer assets. Alternate consolidated
exports were not downloaded or merged. Voxtral checkpoint terms are Apache-2.0;
original VibeVoice checkpoint/source terms are MIT. Nemotron's terms are
OpenMDW-1.1. Publisher terms/notices remain separate from repository licensing.
Source pins for Nemotron/Voxtral use official Transformers
`469230357aab0f2b303b0d638c1f8d06edb14184`; VibeVoice uses the recovered
original Microsoft source below. Further reference dependency locks and actual
parity tests for later families remain WIP. Do not infer an available `speak`,
Nemotron or Voxtral inference path from a completed contract.
The original VibeVoice TTS source was recovered from an archive at commit
`6065c5224e1fee5b2eaeef15f79ee8ef7de947d0`, dated 2025-09-02, and verified
through [Microsoft's original commit API](https://api.github.com/repos/microsoft/VibeVoice/commits/6065c5224e1fee5b2eaeef15f79ee8ef7de947d0).
The publisher commit and archive have the same Git tree
`fb03d34019e571e4be18298fa2b5437b5f5a4e01`. This verifies source identity;
it is not a claim of a signed commit or permission from recorded speakers.
The original processor explicitly requires `Qwen/Qwen2.5-1.5B` tokenizer assets,
separately pinned to `8faed761d45a263340a0528343f099c05c9a4323`.

The [source/tokenizer contract](../qualification/results/vibevoice-original-source-tokenizer-contract.json)
records source file, license, tokenizer and public voice-file checksums; all
voice-fixture permission evidence is marked **WIP**, never inferred from MIT
licensing or public availability. The preserved original README warns against
misleading impersonation/disinformation, recommends AI disclosure and describes
research/development use. Retain these notices when porting. Source and asset
inspection did not execute original TTS code. Native synthesis, processor/diffusion
parity, reference dependency locks and consent/intended-use evidence remain WIP.

Regression validation on this update: `541 passed, 22 skipped` network-free
suite; `30 passed` selected
Metal audio regressions. Documentation checks covered 27 registry types and 36
Python examples. These validation counts are not checkpoint quality gates.

The stage-four–six pending entries are not advertised as supported loaders or
runnable task APIs. The generic `mlx_one.audio.load_audio_model` loads Whisper,
original Moonshine and Qwen ASR/alignment; unavailable families/tasks are rejected
before Metal initialization. Qwen task resolution uses the nested thinker config:
ForcedAligner advertises alignment only, never transcription.

Long-form speech, additional datasets/speakers, translation and non-English
Distil-Whisper quality are outside this measured scope. The fixture selection is
small and ordered, not a representative multi-speaker production evaluation.
