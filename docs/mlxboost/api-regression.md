# API, packaging and regression evidence

Final evidence is under `local/mlxboost/runs/integrity-20261101/`. Run IDs retain
the campaign start date; execution timestamps in the artifacts are authoritative.

| Check | Result / scope |
| --- | --- |
| New backend-free tests | 20 passed; 6 native cases skipped in this invocation |
| Explicit Metal primitive/cache/sampling tests | 6 passed |
| Regression remainder | 2191 passed, 79 skipped, 9 subtests passed; one legacy hybrid case executed separately |
| Isolated legacy hybrid case | 1 passed |
| Monolithic suite | BLOCKED by an inherited order-sensitive hybrid hang, reproduced on unmodified HEAD |
| Authentication and invalid input | Real TCP PASS |
| Stream/non-stream exact bytes and usage | Real TCP PASS |
| Subtoken stop | Real TCP PASS |
| Decode disconnect and subsequent request recovery | Real TCP PASS |
| 100 repeated B1 requests | Exact bytes/counts/cache-off PASS |
| Mid-prefill cancellation / network backpressure | NOT MEASURED as real-network scenarios; existing unit coverage retained |
| Fresh wheel build/install, CLI and generation | PASS |
| Clean wheel dependency check | PASS; MLX/Metal 0.32.2, no MLX-LM/oMLX/Transformers installed |
| Installed runtime import and wheel source/metadata scan | PASS |

The full regression remainder and isolated case cover 2192 passing cases across
fresh processes. This is not represented as a monolithic-suite PASS. The original
full invocation stalled in
`test_existing_affine_int4_hybrid_omitted_policy_is_preserved`. A bounded audit
loaded source extracted from unchanged commit
`c0bcfba3b0d19cd50a6b547ab170e89592083438` and reproduced the same stall at
`tests/test_p1_runtime.py:548`; the traceback and 90-second timeout are preserved
in `baseline-regression.{log,json}`. That unrelated hybrid path and its test were
not rewritten. The first collection attempt also found a new test module name
collision with OCR; adding `tests/mlxboost/__init__.py` fixed collection.
A sandboxed retry aborted when existing native tests imported MLX without Metal;
subsequent execution used the host Metal context. All earlier failures remain.

Real API verification uses the existing server, cached Qwen2.5 Q8 checkpoint,
explicit B1 mode, greedy output and disabled cache reuse. The final repeated run
uses the retained client environment with its RSS sampler:
`api-integrity-client-repeat/results.json`. All seven checks passed. MLX active
memory at every terminal metric was exactly **530493192 bytes** across 100
identical requests. Sampled server peak RSS was **734232576 bytes**, including
loading and warmup. These are different memory measurements. This bounded run
does not establish a long-soak leak guarantee. The initial API invocation also
passed functional checks but lacked the client RSS dependency; its sampler error
is retained and its RSS is not treated as measured.

The fresh environment uses the original native-only server dependency lock and
the newly built wheel, without modifying the development or competitor
environments. It ran eight-token generation through `mlxboost_b1`, checked the
resolved feature selection, loaded the packaged source manifest and verified
that none of `mlx_lm`, `omlx`, or `transformers` was imported or installed.
The wheel AST scan found no imports of those runtimes; metadata has no mandatory
dependency on them. The pre-existing optional `legacy-mlx-lm` extra remains.

Wheel: `mlx_one-0.1.0a1-py3-none-any.whl`.
SHA-256: `d05a98808550447a37085dcf53510dfc3bb8ff4bd0594c038a0fd6b239bab3d3`.
The wheel contains Python/data and relies on the existing pinned MLX packages;
no custom Metal extension or new build ABI was introduced. Verified platform:
Python 3.12.11, Apple M4/32 GiB, macOS 15.8.1, MLX/Metal 0.32.2. Other Python,
OS and device combinations are not qualified. MLX 0.32.4 feature fallback is
unit-tested, not presented as an optimized-runtime qualification.

New runtime/benchmark/test modules pass Ruff; changed core modules pass Ruff.
Existing CLI lint outside the changed lines is not silently reformatted. No
numerical tolerance, frozen test, original benchmark, checkpoint or historical
evidence was edited. The owner's pre-existing pyproject description change was
preserved. No commit, release, publish or deployment was performed.
