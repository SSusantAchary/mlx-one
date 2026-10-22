# Implementation record

Phase 0 changes:

- `native_correctness.py`: explicit backend prerequisites, status schema including
  legacy BLOCKED, checkpoint hashing, exact test nodes/commands, artifact validation.
- `native_qualification.py`: selected backend manifests and correctness cohorts,
  independent failure recording, explicit paged deferral, five-workload Phase 0
  manifest, fail-closed performance exit status, configured watchdog ceiling.
- `inference_qualification.write_integrity`: include logs and JUnit XML.
- `test_p1_checkpoint_parity.py`: attach actual checkpoint and test identities to
  all existing six-checkpoint parity cases without reducing their coverage.
- `BatchedDenseRunner.execute`: count actual logical history payload copied into
  packing and scatter separately from padded allocations; report pack-graph,
  evaluated forward, evaluated staging and commit elapsed time. No additional
  device synchronization or changes to model arithmetic/cache ownership.
- `scripts/qualification/`: executable, offline smoke/correctness/performance/full
  commands, bounded child workers, isolated process-group cleanup and durable
  JSON/CSV/log/Markdown output. Missing local checkpoints are rejected.
- `BaselineProfile.exact_prompt_tokens`, `build_exact_messages`, replay and
  `p1_decode.native_decode`: the new Phase 0 workload checks rendered token
  counts including template/suffix. Early EOS fails the requested output-length
  protocol instead of being reported as a 128-token measurement. Older workload
  defaults and profile identifiers retain their existing behavior.

Copy bytes describe tensor payloads represented by copy operations. They do not
claim measured GPU bandwidth, physical traffic or zero-copy execution. GPU-only
synchronization time remains null because existing wall-clock intervals do not
isolate it from host work.

The v2 correctness schema explicitly rejects v1 receipts for new admission;
historical files remain untouched/readable. New protocol digests cover the complete
test tree and qualification scripts. No existing tolerance or performance gate was relaxed.

Phase 1 changes:

- `ops/attention.validate_attention_inputs`: validate positive Q/K/V dimensions,
  GQA ratio, batch/head/length consistency and actual mask broadcasting before
  cache append, including reference/fallback policies.
- `ops/attention.attention_fused`: shared implementation using MLX force_fused,
  explicit Metal/version/dtype/head checks, native KV-head shapes, FP32 or native
  diagnostic operands, and no retry after failure. Existing runtime experiments
  reuse its FP32 route; their fallback decisions are unchanged. No new automatic
  or production-qualified route is introduced.
- `test_sdpa_gqa.py`: B1/B2/B4 MHA/GQA/MQA forced-fused CPU doubles, no KV repeat,
  invalid reference inputs, native-dtype diagnostics, precision preservation,
  device rejection and failed-Qwen fallback coverage.
- `test_fused_attention_native.py`: 223 operator cases covering FP16/BF16,
  MHA/GQA/MQA, B1/B2/B4, decode/full/chunked alignment, causal/boolean/additive
  ragged masks, and 512/2048/4096 prefill at D64/D128. Separate whole-model
  tests exercise shared-attention checkpoints with actual INT8 weights, QK norm,
  RoPE offsets, full/chunked continuation and complete KV state. Hybrid models
  outside that dispatcher are explicitly UNSUPPORTED in this diagnostic scope.
- `fused-*.sh`: supervised M4 diagnostics; performance requires current,
  checksummed, exact-scope correctness and actual dispatch. Results cannot
  authorize any production backend. Kernel-only scope remains independent.

One inherited scheduler test raced cancellation against generator resumption.
Its observation now lives in `finally`, with an additional deterministic
cancellation-at-yield case. Cancellation and subsequent failure isolation remain
asserted. No scheduler production code changed. The failed run is preserved in
`baseline/final-pytest.log`, followed by the repair's regression logs.
