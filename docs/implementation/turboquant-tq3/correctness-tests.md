# Correctness and functional evidence

Required packed arithmetic is tested independently of the retired numerical research gate.
The NumPy oracle is diagnostic-only in `benchmarks/turboquant/oracle.py`; production never imports it.
It packs and reconstructs small matrices with independent scalar column indexing and FP64 arithmetic.
A Sylvester matrix supplies an independent power-of-two rotation and inverse.

`tests/models/test_turboquant_contracts.py` covers all eight codes, first/last word slots, high unused
bits, final padding, K=128 tails, scale boundaries at columns 63/64, orientation/expert banks,
actual pinned header/index/dtype agreement, invalid metadata, format labels and dispatch boundaries.

`tests/models/test_turboquant_native.py` covers GEMV/GEMM at rows 1/2/8/16/32/64/128/256,
K=64/128/4096, output tails, FP16/FP32, MLX dense and independent NumPy comparisons;
shared/per-route and sorted/unsorted gather, including large prefill. Reference-generated fixtures
at K=64/2880/4096 verify stored signs and the real GPT block rotation. A complete tiny GPT is compared
with an independently reconstructed dense model, and uses top four experts throughout.
Tiny safetensors tests cover repeated loading, first use on the Engine owner thread, cache prefill,
continuation/reset/isolation, FP16 storage, finite logits and malformed parameters. Existing GPT tests
continue to cover sinks, window boundaries, YaRN, activation math and geometry.

Python/CLI/authenticated HTTP/chat/SSE tests execute actual packed forwards with a scripted sampler
so Harmony analysis/final and multi-EOS behavior are observable. Cancellation, recovery, active-handle
cleanup and unload are checked. Scripted synthetic text is not model-quality evidence.

Tolerances: FP32 packed-dot checks use absolute/relative 2e-5; FP16 checks allow output rounding
(absolute up to .002 for the longest sums, relative .001). FP32 accumulation is compared against
FP64 reconstruction on the actually rounded rotated activations. Small gather checks use absolute
.0005 / relative .001. Reference-generated FP16 rotation checks allow .008 / .002 because the
reference's small-output prefill fallback rounds reconstructed weights to FP16 before GEMM.
The independent full tiny-model reconstruction passes .0001 / .001. These limits address arithmetic
precision; they are not language-model E/KL/NLL admission thresholds.

Final regressions:

| Check | Result |
|---|---|
| New TQ3 contract/native cases | 87 focused passing; 86 included in full regressions |
| Default suite | 1978 passed, 68 skipped, 3 preexisting warnings |
| Metal suite (`MLX_ONE_RUN_MLX_TESTS=1`) | 2244 passed, 18 skipped, 3 warnings |
| Ruff debt gate | 65 existing findings, zero new |
| New runtime/tests/benchmarks lint | PASS |
| Wheel and sdist source/private-payload/license audit | PASS |
| Fresh wheel dependency check | PASS, no broken requirements |
| Fresh wheel TQ3 Python/CLI/HTTP/SSE/auth/cancel/unload | PASS |
| Real CLI server / TCP chat+SSE / disconnect recovery | PASS |

Commands: `.venv/bin/python -m pytest -q --no-cov`; repeat with
`MLX_ONE_RUN_MLX_TESTS=1`. Focused tests are the two `tests/models/test_turboquant_*.py` files.
Raw regression, wheel and benchmark evidence is retained in `evidence/`.
The full 20B checkpoint was not downloaded or executed.
