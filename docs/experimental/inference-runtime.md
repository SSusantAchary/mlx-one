# Inference runtime experiments

[Experimental index](README.md) · [Known limitations](known-limitations.md)

## Native inference engine

mlx-one now exposes a lifecycle-managed Python engine while retaining the
existing `generate()` and `stream_generate()` functions:

```python
from mlx_one import Engine

with Engine(model="LiquidAI/LFM2-350M") as engine:
    result = engine.generate("Explain unified memory briefly.", max_tokens=128)
    print(result.text)

    for chunk in engine.stream("Name two advantages of MLX."):
        print(chunk.text, end="", flush=True)
```

The runtime includes typed sequence/runner contracts, isolated request RNG,
bounded output queues, transactional caches, byte-bounded prefix sharing and
explicit execution modes. HTTP prefix namespaces derive from authenticated
principals; anonymous requests remain private. Python callers can supply an
explicit cache namespace.

| Mode | Execution and availability |
|---|---|
| `reference_dense` | Existing single-request dense execution; cooperative server slots use separate B=1 forwards. |
| `experimental_block` | Materialized block-cache path for supported unquantized Qwen2/Qwen2.5; experimental. |
| `experimental_dense_batch` | Actor-driven dense batching for supported native adapters; experimental and not production-qualified. |
| `paged` | Direct page-backed attention using the segmented arena and existing Metal kernels; supported contracts only, qualification pending. |
| `experimental_hybrid_batch` | Native LFM2 attention and request-local convolution state; qualification pending. |
| `auto` | Opt-in selection from matching performance-qualified evidence; otherwise reference dense with a reported reason. |

Omitting the mode preserves existing behavior. The default active-sequence limit
remains one; larger batches require explicit configuration. Unsupported explicit
optimized modes raise capability errors. Native affine INT8 loading and floating
FP16 conversion preserve packed weight payloads; optimized P1 profiles use FP16 KV.
Quantized paged KV and 128K-class admission remain evidence-gated.

See the [repository summary](../REPOSITORY_SUMMARY.md) for current validation,
layout and benchmark status, and the
[native inference-engine roadmap](../architecture/inference-engine.md) for broader plans.

## Cache runtime qualification

Cache and optimized execution paths remain experimental; dense B1 is the bounded
reference scope. Historical P1 results do not certify other checkpoints, longer
contexts or optimized backends. Model admission now follows the
[functional support policy](../model-admission.md).

For the recorded workloads, numerical investigations, limitations and development
commands, see the [P1 execution report](../p1_execution_report.md),
[P1 qualification matrix](../p1_qualification_matrix.md) and
[M4 cache report](../cache-benchmark-m4-32gb.md).

## Experiment map

| Subject | Current source/evidence | Boundary |
| --- | --- | --- |
| Dense, quantized KV and block caches | [Cache Runtime V1](../cache-runtime.md), [cache qualification](../cache-qualification.md) | Dense remains default; block Qwen2/Qwen2.5 B1 f16 scope is opt-in |
| SSD prefix snapshots | [Cold-prefix store](../../src/mlx_one/engine/cold_prefix.py), [cache-manager integration](../../src/mlx_one/engine/cache_manager.py) | Opt-in byte-bounded on-disk snapshots and bounded restore staging; not active-request SSD paging |
| Prefix reuse and ownership | [Cache report](../cache-benchmark-m4-32gb.md) | Dense divergent-suffix reuse was not demonstrated; hybrid prefix restore is unsafe and rejected |
| Actor, token budgets, chunked prefill, dense/hybrid batch and paged runner contracts | [Native baseline](../native-inference-baseline.md), [P1 execution](../p1_execution_report.md) | Source components exist; production continuous batching is not established |
| Attention, last-token projection and quantized computation investigations | [P1 specification](../p1_implementation_spec.md), [failure log](../p1_native_failure_log.md), [optimization records](../inference_optimization) | Native optimized mode has explicit capability checks; unavailable specializations reject |
| Memory pressure and context admission | [P1 hardware report](../p1_native_hardware_report.md), [cache benchmark](../cache-benchmark-m4-32gb.md) | Budget and host specific; configured position limits are not measured context capacity |
| Numerical/reference gaps | [P1 reference gaps](../p1_reference_gap_analysis.md), [qualification matrix](../p1_qualification_matrix.md) | Historical failures are preserved, independent of current functional admission |

The M4/32 GB unquantized Qwen2.5-0.5B dense experiment completed 16K input tokens
but failed around 32K at the recorded budget. Dense snapshot APC did not reuse
divergent chat-prompt suffixes. Block sharing, APC and dense/block parity remain
unqualified by that report; LFM hybrid block/APC combinations reject explicitly.

## Current components versus plans

Earlier Cache Runtime V1 documentation describes an earlier scheduler stage;
the current tree contains `InferenceActor`, `BatchScheduler`, native paged/dense/
hybrid runner contracts, prefix/page ownership and live memory-pressure hooks.
Their presence does not establish production continuous batching or universal
paged attention support. Matching evidence controls optimized selection; the
reference single-sequence path remains the default.

Active-request SSD paging and generalized production continuous serving remain
roadmap work. The implemented optional SSD cold-prefix store persists immutable
safetensors snapshots; it does not page live attention state through SSD. Existing Metal
attention code does not imply the proposed MLXBoost custom kernels shipped.
Qwen3.5 MTP options are bounded existing functionality, not general speculative
model coverage. See the [engine roadmap](../architecture/inference-engine.md)
and [MLXBoost proposal status](mlxboost.md).
