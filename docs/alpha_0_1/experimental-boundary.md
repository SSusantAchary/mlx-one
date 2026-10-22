# A1 experimental boundary

VERIFIED_NO_CHANGE to execution source. The current `Engine`/construction/modes and request-scoped NativeMathPolicy already protect reference execution; moving code would add import/cache ownership risk.

| Existing implementation | Selection / status | Preserved boundary |
|---|---|---|
| reference_dense, reference attention/projection | Default dense; reference math | Existing engine/cache/transactions unchanged |
| native_optimized / last_only | Explicit B1 FP16 mode; unqualified | Reference-mode operator overrides rejected before weight load |
| native_gather_tiled / reference_tiled | Explicit math policy; exact MLX/chip/layout guard | Scoped prior feasibility evidence only; no alpha promotion |
| mlx_sdpa / forced fused | Explicit diagnostics; known state parity failures | Qwen and decode guards keep reference fallback with reason |
| paged / hybrid / dense batch / block | Explicit mode and model/precision/layout checks | No physical moves or scheduler replacement |
| auto | Explicit auto selection with evidence scope | Existing behavior preserved; no widening |
| quantized/cold prefix, speculation, MTP, compile | Existing opt-in / evidence gates | Prior failures and unimplemented specialization errors preserved |
| default dense prefix reuse | Existing cache behavior, enabled in CacheConfig | Namespace/ownership/lifecycle tests retained; qualification command must explicitly disable reuse when measuring no-prefix execution |

No new experimental namespace, shims or duplicate implementations. Import smoke passed without Metal or model loads. Historical numerical failures remain untouched and source-bound.

Focused verification: `PYTHONPATH=src:tests:. .venv/bin/python -m pytest --no-cov -q tests/optimization/test_math_policy.py tests/optimization/test_engine_math.py tests/optimization/test_native_gather_dispatch.py tests/test_cache_runtime.py tests/test_state_transactions.py tests/test_device_executor.py tests/test_engine_public.py` — **79 passed**, 1.18s. Includes invalid policy, negative capabilities, exception/cancellation cleanup, request policy isolation and cache transactions. Evidence: ignored session `a1.log`, `a1.junit.xml`. This is backend-free evidence, not fresh numerical checkpoint proof. `git diff --check` PASS.
