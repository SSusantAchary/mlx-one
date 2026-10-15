# Registry native inference: PR-11–PR-20 migration

These commits add task integration and opt-in experiments. They do not establish
production qualification for all registered architectures. Existing generation
signatures, HTTP responses, SSE framing, omitted-mode behavior, authenticated
namespaces and default active-sequence limits are preserved.

## Task services

```python
from mlx_one import BatchingConfig
from mlx_one.engine.service import InferenceService

service = InferenceService(batching_config=BatchingConfig(max_sequences=4))
service.register_task("encoder", "/local/pinned/embedding-checkpoint", offline=True)
result = service.embed("encoder", ["query", "document"])
print(service.task_capabilities("encoder", "embedding"))
service.close()
```

`submit_task` returns a cancellable handle; `run_task` returns a request-local
`TaskResult`. Task concurrency, padded-input costs and transient bytes are bounded.
Larger images may need an explicit `task_transient_budget_bytes` allowance; the
default remains 256 MiB. Actual forward batch sizes are measured separately from
queued requests. Retrieval rows can batch; static-image VLM and complete
Whisper/Moonshine/Qwen transcript bridges retain singleton reference forwards.
Ready Whisper/Moonshine decoder components and Parakeet TDT rows have native B2
coverage. End-to-end heterogeneous speech batching remains incomplete.

Speech tasks accept copied mono PCM with explicit `sample_rate=16000`. Encoded
file decoding remains available through existing audio APIs. Qwen utterance
sessions use `open_audio_session` and ordered `session.call("push", ...)`,
`finish`, and `reset`. They reprocess accumulated audio; they are not cached
streaming encoders. Alignment is separate from transcription. Its inherited
800-ms discrepancy against an 80-ms gate remains failed qualification.

Static VLM tasks accept a prompt with matching `<image>` placeholders and local
images. Complete request-owned caches preserve multimodal positions and carries.
Qwen3.5 keeps separate text and image adapters. LFM2-VL currently accepts only the
explicit `lfm2-vl-v1` native tensor format and native processor contract; publisher
conversion is not implemented. Video/audio VLM inputs are rejected.

SAM3 image workflows preserve float32. `open_video_session` retains an epoch
lease and ordered frame/object state until `close`. Nested `sam3_tracker_video`
is a component, not an independently loadable service. No SAM3/VLM GPU batching
or checkpoint mask/processor qualification is claimed.

LFM2 Audio remains architecture-only: safe weights, exact processor/tokenizer,
codec cadence and flush contracts are missing. No ASR/TTS/speech-to-speech task
is inferred from its component classes.

## Optional storage and speculation

SSD prefixes are disabled unless `CacheConfig.ssd_cache_directory` and a positive
`ssd_cache_budget_bytes` are provided. Use a dedicated directory; set explicit
`ssd_staging_budget_bytes` and `ssd_max_entries` as needed. Files use immutable
safetensors generations and atomic checksummed JSON indexes. Runtime close
releases RAM owners while preserving the SSD tier. Private namespaces, complete
hybrid snapshots and one-token re-execution remain enforced. Current manager
integration supports FP16; quantized KV and multimodal prefixes are excluded.
Restoration reserves staging separately from resident/cache storage.

```python
from mlx_one import Engine

# A complete Qwen3.5 MTP checkpoint, greedy reference execution, one active request.
engine = Engine(bundle, inference_mode="reference_dense", spec_type="draft-mtp")

# Explicit native draft bundle with identical serialized tokenizer/EOS semantics.
engine = Engine(target_bundle, inference_mode="experimental_dense_batch",
                execution_dtype="float16", spec_type="draft-model",
                draft_model=draft_bundle,
                batching_config=BatchingConfig(max_sequences=2))
```

External drafting is Python-only and greedy dense. Target verification checks all
real positions in actual batched forwards; draft state currently rebuilds its
private history each round. Speed benefit is unmeasured. Unsupported modes fail
explicitly. There is no streamed replay or automatic speculative fallback.

## Qualification commands

```sh
PYTHONPATH=src python benchmarks/native_tasks.py inventory
PYTHONPATH=src python benchmarks/native_tasks.py plan --profile /local/task-profile.json
PYTHONPATH=src python benchmarks/native_tasks.py run --profile /local/task-profile.json
PYTHONPATH=src python benchmarks/native_tasks.py matrix --output /local/task-matrix /local/run
```

Profiles use `native-task-profile-v1`, pinned revisions, local checkpoint paths,
a checksummed `native-task-inputs-v1` manifest, explicit limits and one warm-up.
Text benchmarks require an explicit resolved inference mode and reject `auto`.
Five repetitions are the performance minimum. Every input media asset requires
a checksum; audio fixtures use decoded mono `.npy` PCM without implicit resampling.
Commands never download assets. Run native workers with Metal/RSS permissions;
the existing preflight and watchdog cannot be disabled through this harness.

Artifacts remain local under `qualification/results/<host>/<checkpoint>/<run>/`.
They distinguish native task completion throughput from HTTP service throughput,
model-token gaps and SSE timing. Unmeasured fields remain null. Continue using
the existing P1 HTTP/traffic profiles for those service measurements. Task corpus
coverage and full lifecycle/traffic gates are pending when their evidence is
absent. Synthetic smoke never unlocks public capabilities, `auto`, quantized KV
or long-context admission. Task selection requires an exact current qualified
profile through the existing feature registry; defaults are unchanged.

## Adding a local architecture

Keep configuration, task specifications, state metadata and loader modules
backend-free. Put native MLX math in the model module. Supply both a
`ModelRegistration` and a versioned `RuntimeRegistration`, including per-task
adapter bindings where one architecture supports multiple tasks.

`mlx_one.engine.plugins.validate_plugin(..., config=sample_config)` validates
selected metadata in an isolated process that rejects MLX/Torch imports.
`register_plugin` publishes architecture and runtime descriptors atomically and
refuses replacements. No scheduler edits are needed for the synthetic encoder
example in `tests/registry_plugin_fixture.py`. That example is CPU test code,
not a native or performance-qualified implementation.

Metadata validation establishes neither checkpoint compatibility nor hardware
qualification. New plugins must provide safe tensor contracts, task correctness,
state isolation, cancellation and measured service evidence before rollout.
The inherited Qwen3-small INT8 paged B2/page16 parity failure remains failed;
these commits do not repair or promote that backend.
