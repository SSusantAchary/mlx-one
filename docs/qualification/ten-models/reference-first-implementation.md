> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Reference-first native implementation

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Source pin: `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902` plus this cohort's frozen dirty entry and final file hashes.

The default GPT-2, Qwen2, Qwen3, Llama, LFM2 and Qwen3.5 text paths now execute MLX-One-owned reference mathematics. This replaces their former shared explicit reductions, gathered affine projection policy and independently written activation order. Qwen2 is shared by both exact Qwen2.5 sizes; Llama is shared by SmolLM2 and Llama3.2. The previous explicit Qwen3 upstream candidate remains available and unchanged.

Gemma3 and Phi3 matching operations are reused, as required by the brief. Phi3's MLP now calls the source's compiled SwiGLU. No mathematical rewrite is claimed for already matching Gemma3 code. Both are covered by fresh full/cache/chunk/reset and independent window/attention controls.

Ported attention uses the source Q/K/V construction and native SDPA. Ported linears use stock MLX modules, including packed QuantizedLinear/QuantizedEmbedding. A small adapter calls MLX-One cache.update instead of update_and_fetch, dispatches existing paged-cache attend when supplied, propagates positions and preserves padding. Cache ownership, reservations, scheduling, serving, public loading and the 29 registrations remain MLX-One's.

Qwen3.5 retains raw offset norm weights and HF's FP32 normalization/weight-addition order. The reference folds norm weights in checkpoint precision; this intentional difference remains a separate numerical gate. Its fixed convolution accumulates products in FP32 before output rounding. Ported decay promotes A_log and a before exponentiation/softplus. The existing [B,H,K,V] recurrent state owner is retained; the ported helper uses [B,H,V,K] with explicit contiguous boundary transposes. The source gate normalization and recurrent kernel helpers are adopted. Vision and MTP remain available but are outside this text-only mathematical qualification.

Compiled activation and recurrent helpers load on first mathematical use, preserving backend-free metadata and model-boundary tooling. Superseded shared primitives remain for other registered architectures and experimental modules. LFM2 MoE/multimodal subclasses retain their previous layer contract; this is an explicit adapter branch, not a rewrite of their mathematics.

Every whole-port test runs without Engine/scheduler/HTTP, using matched parameters, full input, five partitions, incremental tokens, state offsets and reset. Independent NumPy controls validate attention and recurrent equations. Real source checkpoints additionally execute all 17 forced-continuation boundaries, committed state comparisons, fresh-prefix lengths 1..64, requested lengths 1/2/3/7/15/16/17/32/64 and six chunk partitions. Public API checks follow separately. A matching reference with failed self-consistency cannot qualify a checkpoint.

See [mathematical port diff](../archived-evidence.md), [source provenance](reference-source-provenance.md), and [final matrix](final-10-model-matrix.md).

Prefix fingerprints now include `reference-first-v1` for these native models, including stored-weight execution without an explicit floating policy. Qwen3.5 recurrent snapshots use `qwen35-gated-delta-v2-reference-first`; previous transition snapshots are rejected. The Engine/cache ownership is retained. Targeted tests cover the identity change and version rejection; no existing stored states are deleted.

The exact changed-file list and production diff are retained in [reference-first-20261020-1792502347743318000/changed-files.md](../archived-evidence.md) and [reference-first-20261020-1792502347743318000/production-change.patch](../archived-evidence.md). Entry cache implementations fail both new identity/version tests, while the finalized implementation passes; see [reference-first-20261020-1792502347743318000/cache-contracts-entry-failing.log](../archived-evidence.md).

The newly ported Llama3RoPE class is also checked directly at offsets 0/4096/8192/32768 and factors 8/32 across 64 dimensions against the local source; its strengthened whole-port test passes. The preexisting ordinary/three-band shared-helper regressions remain intact.
