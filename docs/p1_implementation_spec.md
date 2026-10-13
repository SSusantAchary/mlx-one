# P1 implementation specification

Base `6d2c9a1bbc36f1972b468ca841977e3b342fbb36`; ordered boundaries PR-00–08.
Use the existing actor, scheduler, cache owner, arena, dense/paged/hybrid runners.
Qwen3-0.6B INT8 dense comes first, then paged Qwen3, SmolLM2 and LFM2.

INT8 means native MLX affine packed weights, initially group 64. Validate actual
module/tensor metadata. FP16 execution is explicit: convert floating components
only, preserve integer payloads, and include conversion identity in cache/evidence.
FP32 accumulation is allowed. Omitted-policy reference behavior is preserved.
Adapters own architecture/topology/positions/normalization and eligibility; generic
scheduling and ownership must not branch on model family. Preserve explicit Qwen3
head_dim=128. Prefix identities include quantization and floating execution policy.

Initial profiles: B1/B2/B4; prompt 128/512/2048/8192; output 32/128/256;
chunks 128/256/512/1024; pages 16/32. Defaults remain chunk 512, page 32,
one active sequence, and reference execution. SmolLM2's 8192 total context includes
output, so oversized combinations reject before load; maximum-legal prompts are
separate named profiles. Preserve request-local RNG, stops, masks and carry state.

Explicit optimized modes fail unsupported selections. Opt-in auto selects before
pool construction, freezes per epoch, and requires fresh matching evidence.
LFM2 dense hybrid uses experimental_hybrid_batch; paged LFM2 pages attention only.
CPU readiness cannot promote hardware capabilities. No SSD persistence, quantized
KV, speculative decoding, multimodal expansion or distributed inference.
