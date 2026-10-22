# Native GPT-OSS TQ3 implementation

Completed in the existing `gpt_oss` architecture and native Engine/ModelManager/scheduler.
The target is `manjunathshiva/gpt-oss-20b-tq3` at revision
`20ba1835e625264e07366d215933fe31a9940a6a`. Full weights were not downloaded.

1. Pin reference source and retrieve only JSON/tokenizer metadata and bounded tensor headers.
2. Establish independent NumPy packing and reconstruction tests before optimized execution.
3. Add strict typed TQ3 metadata, compressed layer creation and exact parameter validation.
4. Implement fused packed dense and routed expert arithmetic, including tiled prefill.
5. Validate tiny native GPT, cache ownership, Harmony, Python/CLI/HTTP/SSE and cancellation.
6. Measure baseline, dispatch alternatives and complete synthetic decoder blocks; reject tuning without broad gains.
7. Run default/Metal regressions and install/audit a clean wheel with references absent.
8. Hand off real checkpoint validation and fair full-model benchmarks to the owner.

No second architecture registry entry, inference engine, scheduler, tokenizer wrapper, KV compressor,
paging implementation or production MLX-LM/TurboQuant dependency was added.

See [checkpoint-format](checkpoint-format.md), [kernel-design](kernel-design.md),
[correctness-tests](correctness-tests.md), [performance-report](performance-report.md),
and [manual-validation](manual-validation.md). Source pins and SHA-256 hashes are in
[source-provenance.json](source-provenance.json).
