# Release readiness

Current candidate status: **QUANTIZATION_BACKEND_TESTED**.
Architecture: **ARCHITECTURE_IMPLEMENTED**, native `gpt_oss` (same registry entry as MXFP4-Q4).
Checkpoint: **NOT_DOWNLOADED_NOT_SMOKE_TESTED**. Owner validation and full-model performance:
**PENDING_USER_DOWNLOAD**. Neither EXPERIMENTAL_SUPPORTED nor USER_VALIDATED is granted.
Alpha publication approval remains unchanged and pending.

Ready and tested: strict original compressed TQ3 loader, fused packed GEMV, tiled packed GEMM,
selected expert GEMV/gather GEMM, floating/explicit mixed exceptions, native GPT architecture,
FP16 full/sliding KV, Engine and manager ownership, Python/CLI/authenticated HTTP/chat/SSE,
cancellation/recovery, strict malformed-storage failures, clean-wheel installation and notices.
The existing MXFP4-Q4 GPT candidate and its distinct implementation remain intact.

Remaining evidence requires the owner's full checkpoint: actual load/visible final generation,
real API quality and resource cleanup, practical context/memory limits, same-host reference/native
warm decode and prefill. Begin with 128 or 512 input tokens, 128 output budget; allow EOS to stop
naturally. Attempt 2048/8192/16384 only when the benchmark's conservative memory/context preflight
allows it, and record skipped workloads. No supported-context or >=90% performance claim follows
from configuration, synthetic blocks or a short real-model smoke alone.

Only promote after the actual checkpoint passes the documented functional policy. The historical
research E/KL/FP64 suite remains optional and is not a model-admission gate. Preserve exact revision,
weight hashes, prompt IDs, software versions, output counts and backend settings in the evidence.

See [manual-validation](manual-validation.md) for the handoff and [verification.json](../../qualification/archived-evidence.md)
for source/artifact/evidence hashes. Other user changes in the worktree were preserved; no commit,
release, push, model download or publication was performed.
