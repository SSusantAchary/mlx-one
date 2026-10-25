# Native audio implementation report — 2026-10-24

**Native source implementation delivered; real-checkpoint acceptance remains manual.**
LFM now has native ASR/TTS/speech-to-speech and an actual waveform codec path.
Required Qwen Q8 and MLX Parakeet formats have explicit loaders. Qwen alignment
and original NeMo validation retain the WIP limitations below. No large audio
model weights were downloaded, no real audio benchmark scores were fabricated,
and no retired exhaustive numerical campaign was run.

The [machine-readable report](report.json) records identities, per-family initial
gaps, exact final test counts, artifacts, limitations and changed files.
[source-files.json](source-files.json) hashes current task-related files, including
shared files which already contained earlier VLM changes. Those earlier changes
and unrelated working-tree research files were preserved.

1. **Branch and base.** Work remains uncommitted on
   `chore/alpha-repository-cleanup`, base
   `eda7a0b23ecfc14b90a5af28f84dff70b4d7629a`. No commit, push, tag or release.
   This was an already dirty checkout, not an isolated clean-source release.

2. **Initial inventory and baseline.** The native architecture/config/loader/
   frontend/decoding modules, model/task registries, existing inference actor,
   model pool, audio services/sessions, CLI, Whisper HTTP sidecar, conversion tools,
   qualification reports and targeted tests were inspected before edits.
   The pre-edit audio baseline was **156 passed, 5 skipped**;
   [baseline.txt](baseline.txt) retains its log. `report.json` records six family
   inventories. Whisper/Moonshine/Qwen/Parakeet already had working native paths;
   LFM had synthetic components and an intentionally unavailable task adapter.

3. **Files changed.** Changes cover the common `audio/` API/input/schema/service,
   task contracts/registration, speech adapters, audio model loaders/processor
   handling, native LFM publisher components, existing CLI/server integration,
   targeted tests/fixtures, package notices and the new benchmark/documentation
   directories. The complete list and hashes are in the JSON manifest. Existing
   text/VLM mathematical modules were not redesigned for audio.

4. **Checkpoint loading.** The manual inventory has **11 unique checkpoints and
   13 task rows**, with immutable revisions and asset selections. Qwen keeps the
   original floating `thinker.*` layout separate from the native MLX packed layout.
   Parakeet maps the MLX export's NeMo-shaped config/native convolution arrays,
   vocabulary and combined LSTM biases into the existing TDT model. LFM uses its
   actual `lfm/encoder/preprocessor/depthformer` schema, not the old synthetic one.
   Unknown, missing, extra or malformed tensors fail strict loading.

5. **Precision.** Existing FP32/FP16/BF16 publisher paths remain. Qwen supports
   native affine Q4/Q8 packed modules; primary manual targets are Q8 with BF16
   exceptions. LFM's primary MLX Q4 export keeps packed linear/embedding tensors
   and floating encoder/convolution/norm exceptions; its separate waveform codec
   retains its own tensors. Parakeet remains the explicitly supported FP32 layout.
   No global FP16 dequantization or unsupported-format reinterpretation is used.

6. **Audio preparation.** New managed inputs support local files, encoded bytes
   with an explicit audio MIME type, and floating/int16 waveforms with a declared
   sample rate. Bounded mixing, finite-value validation, sinc resampling, decoder
   concurrency, temporary-directory cleanup and cancellation precede the family's
   own frontend. Whisper log-Mel, raw Moonshine, Qwen features, Parakeet features
   and LFM features remain distinct task contracts. The official LFM source
   confirms symmetric Hann windows, constant STFT padding, inference dither off,
   per-feature normalization and zeroed padding; targeted native tests cover this.

7. **ASR and language behavior.** Working Whisper, Moonshine, Qwen and Parakeet
   inference mathematics are reused. Whisper retains deterministic long-form
   windowing/context and timestamp-driven advancement; each segment now selects
   its own generated-token attention rows and actual frame range for word
   alignment, excluding prefix/EOS rows. Whisper first-window language detection
   and Qwen's generated label are exposed through managed APIs. English-only or
   unavailable family options fail explicitly. LFM uses its own speech encoder,
   shared native language backbone and `Perform ASR.` template, not another ASR.

8. **Forced alignment.** Alignment stays a distinct audio-plus-known-text task.
   It validates timestamp positions/classes and uses checkpoint tick values and
   existing reference repair. Out-of-clip predictions become explicit unmatched
   words rather than fake confidence or arbitrary clamping. Timeline offsets
   preserve original duration and unmatched words. **The historical up-to-800-ms
   discrepancy against an 80-ms comparison target is not resolved or qualified.**
   Existing evidence localizes one classifier difference upstream of the head;
   no constant-offset/tokenizer workaround is justified without manual fixtures
   and real weights. The 1.7B punctuation score-ordering discrepancy is also
   retained; aggregate matching WER/CER was not rewritten into exact identity.

9. **LFM speech generation.** The precise mapping includes `audio_encoder`,
   `audio_adapter.layers`, the existing `lfm` backbone (depthwise storage rename),
   `audio_embedding`, `depth_linear`, per-codebook `depth_embeddings`,
   `audio_head.depthformer.blocks`, and the separate
   `audio_detokenizer/{config.json,model.safetensors}`. Codebooks are generated
   autoregressively along their depth axis. Real code frames are decoded through
   the loaded waveform model/window with overlap normalization; EOS frames are
   excluded. Codec vocabulary (2,048) is distinguished from the unused text
   embedding vocabulary (65,536). Four publisher voice prompts and interleaved
   speech responses are implemented. Engineered tiny checkpoints execute the
   complete native loader→generation→codec→WAV path; they establish no intelligibility.

10. **Python.** `from mlx_one.audio import AudioEngine, AudioInput` provides managed
    transcription, language detection, alignment, TTS, speech-to-speech and ordered
    streams. Existing low-level APIs and result classes remain available. Typed
    generation results carry actual samples/sample rate/duration and safe WAV/PCM
    writing. [The runbook](../../AUDIO_TESTING.md#python) includes exact examples.

11. **CLI.** `mlx-one audio transcribe`, `detect-language`, `align` and `speak`
    extend the existing executable, with offline loading, clear validation and
    nonzero errors. Existing positional top-level transcription/alignment commands
    remain. Speech output refuses to overwrite files. Model paths load through
    the shared actor, not through a new unmanaged inference loop.

12. **HTTP.** The existing model manager/server/scheduler serves a startup audio
    checkpoint with a strict public alias. Routes cover transcription, applicable
    Whisper translation, language detection, known-text alignment, WAV/PCM speech,
    and a provider-specific speech-to-speech endpoint. Authentication, multipart
    bounds, MIME validation, response formats, task validation, disconnect
    cancellation and structured errors are included. Audio-only chat requests
    receive a capability error; existing text/VLM chat routes remain intact.

13. **Streaming.** Qwen sessions reprocess a bounded accumulated utterance, with
    complete replaceable hypotheses, deterministic chunk boundaries and finalization.
    They are not causal encoder caches. LFM retains native decoder state and emits
    ordered text/decoded PCM with sample offsets. The codec recomputes a bounded
    prefix and withholds the unstable ISTFT tail; it is not a codec KV cache.
    HTTP ASR uses SSE partial/final/completion/error records; generated audio uses
    documented NDJSON frames with base64 signed-16 PCM, never an arbitrary SSE
    text delta or claimed realtime WebSocket protocol.

14. **Ownership and bounds.** The existing inference actor, device executor, model
    pool and server scheduler remain the owners. Model and codec residency are
    accounted together in the managed service. Encoder/decoder/codec/request
    buffers are request-scoped; LFM checks cancellation between encoder layers and
    decoding steps. Model load is retained between requests, and pool leases are
    released after result/error/cancellation/close. Managed input has a 180-second
    ceiling and may hit architecture-specific memory bounds earlier; Qwen streaming
    is 30 seconds per utterance. Direct Whisper local decoding is bounded to ten
    minutes/64 MiB. LFM speech output is capped at 512 codec frames/40.96 seconds;
    exhaustion raises instead of silently truncating.

15. **Final tests.** The default full repository suite passed **2,138 tests,
    71 skipped, 9 subtests** in 33.78 seconds. The affected opt-in native audio/
    shared-LFM/adapter suite passed **61 tests** in 1.07 seconds, including real
    tiny safe-checkpoint ASR/TTS, Q8 execution, BF16 components, strict pinned
    header maps, native HTTP and language detection. Benchmark validator tests
    passed **9**; these overlap the full suite and must not be added to it.
    Final logs: [full suite](full-suite.txt), [native suite](native-suite.txt).
    Intermediate stale registry assertions, an accidentally removed mask helper,
    a wrongly thread-constructed fixture and an alignment benchmark timing-field
    assumption were corrected and retested. Final failures: **zero**. Two existing
    Starlette/httpx/AnyIO deprecation warnings remain. Lint has **65 inherited
    findings, zero new findings**; changed standalone tools/tests pass targeted Ruff.

16. **Packaging.** Wheel and sdist build and pass the payload/source/license audit;
    see [package-audit.json](package-audit.json) and [build.txt](build.txt). The new
    MIT adaptation notice is packaged. A fresh virtual environment installed the
    final wheel outside the checkout. Its smoke test forbids mlx-lm, mlx-audio,
    mlx-whisper, Transformers, PyTorch, NeMo and ONNX runtime imports, then runs
    actual tiny native ASR/TTS, repeats, cancelled-stream disposal, auth/WAV HTTP
    and installed CLI discovery. [Wheel evidence](wheel-smoke.json) reports PASS
    on Apple M4 / 32 GiB / MLX 0.32.2 and zero model residency after close. This is
    package/control-flow evidence using synthetic weights, not publisher inference.

17. **Benchmarks and README.** The reusable local-only runner records actual audio
    durations, load/warmup/repetition protocol, RTF/throughput, WER/CER, memory,
    alignment errors/unmatched words when annotated, Qwen streaming latency/buffer
    metrics, and available LFM input/encoder/codec/first-audio timings. Direct PCM,
    Python Engine and file-pipeline measurements remain distinct. Validation
    rejects missing provenance, fabricated pending values, invalid medians/RTF,
    incompatible cohorts and mixed input fixtures. README updates affect only
    the audio marker region. **All 13 public audio records are NOT_RUN/null.**
    [Coverage](../../benchmarks/audio/coverage.md) and
    [methodology](../../benchmarks/audio/README.md) retain the evidence boundaries.

18. **Manual downloads and commands.** [The catalog](../../../benchmarks/audio/checkpoints.json)
    verifies repository existence, immutable revisions, selected assets and bytes.
    [Per-checkpoint commands](../../benchmarks/audio/manual-commands.md) provide
    downloads, CLI and benchmark invocations for every primary task. A10/A12/A13
    share one LFM download. Original v2 uses the existing isolated safe converter;
    `benchmarks.audio.convert_manifest` validates its provenance/assets before
    benchmark registration. The minimal MLX Whisper export is disclosed as
    incompatible; the primary Tiny entry uses the complete OpenAI assets.

19. **Support classification.** New Qwen packed, MLX Parakeet and LFM checkpoint
    paths are NATIVE_IMPLEMENTED, with no new CHECKPOINT_SMOKE_PASSED or
    EXPERIMENTAL_SUPPORTED promotion from tiny tests. Aligner and original-NeMo
    discrepancies remain WIP. Historical scoped Whisper/Moonshine/Qwen/Parakeet
    evidence remains attached to its own exact revisions and protocols.

20. **Unfinished asset-dependent validation.** Real checkpoint download/inference,
    English/French/German and silent/noisy corpus quality, corrected aligner
    boundaries, original-NeMo parity, LFM speech intelligibility/voice quality and
    standardized hardware benchmarks must be performed manually. No runtime
    correction can truthfully be claimed for the historical classifier/decoder
    disagreements without reproducing the actual fixtures/weights. Moonshine/
    Parakeet causal streaming, long unbounded speech synthesis, Mimi encoding,
    voice cloning and stock OpenAI voices/speed control remain unsupported.

21. **Preservation.** Earlier VLM work and unrelated working-tree files were kept.
    Existing benchmark validators still pass: **17 text measurements across 13
    revisions**, and **40 VLM records including 20 measured records**. Historical
    qualification investigations are linked documents, not reinstated admission
    campaigns. No old report is repackaged into a new standardized audio score.
    No source changes were reverted, no real model caches were deleted, and no
    external publication or repository push was performed.
