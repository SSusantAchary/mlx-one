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

## Implemented contracts

`engine/adapters.py` supplies an immutable native specification, observed affine
module validation, explicit head layout, native forward entry point, component
layout identity and storage estimates. Qwen2, Qwen3 and Llama share the dense and
paged runners. LFM2 uses HybridRunner for both dense hybrid and attention-only
paged execution. Sliding hybrid paging fails explicitly; generic recurrent
transition math remains unavailable.

A paged hybrid prefix owns complete sealed attention pages and copied convolution
carries at one committed boundary. Pins belong to the retained prefix; borrowers
own independent references. Mutable tails stay private. Physical accounting counts
arena storage once and adds carry storage. JSON/tensor lifecycle snapshots use
atomic manifests; explicit diagnostic serialization can export paged payloads,
while generation continues to reject full-history materialization.

Conversion manifests v2 bind saved tensors, tokenizer/config assets, source pin,
converter version and policy to a content-derived revision. SmolLM2 benchmark
workers require verified local converted artifacts and load offline. Explicit
floating conversion leaves packed uint32 objects untouched. Omitted execution
policy preserves the reference path.

Native evidence v4 and workload envelopes v3 supersede earlier diagnostic records.
Adapter, quantization/module map, floating conversion and component layouts bind
selection. Auto resolves before arena construction and remains fixed per epoch.
INT8 performance promotion additionally requires a checksummed P1 gate and matched
native serial/B1 references. P1 targets are never reduced when a run fails.

## Reproduction

Run from the P1 worktree with its source on PYTHONPATH. Profile discovery and local
header inventory do not initialize Metal:

```sh
PYTHONPATH=src python benchmarks/p1_inference.py assets
PYTHONPATH=src python benchmarks/p1_inference.py profiles --checkpoint qwen3-small
PYTHONPATH=src python benchmarks/p1_inference.py download --checkpoint qwen3-small
PYTHONPATH=src python benchmarks/p1_inference.py run --checkpoint qwen3-small \
  --inference-mode experimental_dense_batch --prompt-tokens 2048 \
  --output-tokens 128 --slots 1 2 4 --chunk-tokens 128 256 512 1024
PYTHONPATH=src python benchmarks/p1_inference.py run --checkpoint qwen3-small \
  --inference-mode paged --prompt-tokens 2048 --output-tokens 128 \
  --slots 1 2 4 --page-tokens 16 32
```

Start with the small Qwen3 dense profile on the native host before executing the
larger or paged profiles. Performance comparisons run each configuration in a
fresh offline worker, warm explicitly, and measure five repetitions. Supply
`--serial-run` and `--b1-run` with checksummed matching run directories and declare
`--ttft-budget-ms`, `--decode-gap-budget-ms`, `--max-queue-wait-ms` and
`--max-service-gap-ms`. References must use matched checkpoint, precision, source
and workload. B1 serial profiles use reference_dense and one slot. CPU fixtures
cannot supply these native references.

For SmolLM2, first download its pinned base and run `convert --checkpoint
smol-small --checkpoint-path /absolute/local/artifact`. Use that path in subsequent
profiles/runs. `--max-legal-prompt` labels prompt lengths computed as context minus
output; an 8192-token prompt plus output is rejected. Use the large checkpoint key
only after validating its small counterpart.

Use workloads mixed, turnover, cancellation, slow_consumer, multi_turn, lifecycle
and peak_calibration independently; `--traffic-manifest` accepts the existing
checksummed replay format. For the inherited 8K prefix gate, create a matched run
with `--disable-prefix-reuse` and pass its directory as `--no-prefix-run` to the
prefix run. Absolute limits remain mandatory. Gate results distinguish decode
throughput from end-to-end throughput and model-token gaps from SSE gaps.

Opt-in native tests:

```sh
MLX_ONE_RUN_P1_CHECKPOINTS=1 PYTHONPATH=src:.:tests python -m pytest --no-cov \
  tests/test_p1_checkpoint_parity.py
PYTHONPATH=src python benchmarks/p1_inference.py matrix
```

Set MLX_ONE_P1_SMOL_SMALL_PATH / MLX_ONE_P1_SMOL_LARGE_PATH for converted artifacts.
Native tests load offline, compile the existing paged specializations through real
execution and compare full state. Missing assets, unavailable Metal or unmeasured
service limits leave qualification pending.
