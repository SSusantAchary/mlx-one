> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Upstream vs mlx one cache tests

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Final exact upstream result: 25 tests; 0 failures; 0 errors. The original test definitions and original setUpClass run in a genuinely isolated environment with supported dependencies, using the pinned Qwen1.5 test checkpoint. The source archive lacks generated version metadata; the launcher bypasses package __init__ only, records this shim, and audits all imported mlx_lm module files as local source. Earlier offline-asset and adapted-fixture failures are retained.

These tests cover cache identity, persistence, trimming, rotating/batch bookkeeping, quantized representation and continuation using the same incremental shapes. They do not compare a full model’s GEMM-prefill graph to its GEMV-decode graph at the historical elementwise thresholds. Thus a PASS here does not overturn a canonical failure.

| Upstream test | Result | MLX-One equivalent / scope gap |
|---|---|---|
| test_arrays_cache_advance | PASS | PARTIAL: tests/models/test_alpha_cache_semantics.py::test_lfm_convolution_state_and_output_against_numpy_cross_correlation; no generic ArraysCache extend/advance equivalent |
| test_arrays_cache_extend_with_empty | PASS | PARTIAL: tests/models/test_alpha_cache_semantics.py::test_lfm_convolution_state_and_output_against_numpy_cross_correlation; no generic ArraysCache extend/advance equivalent |
| test_batch_kv_cache | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_attention_caches_batch_select_and_merge; upstream-specific empty/rotating/offset cases MISSING; batching excluded |
| test_batch_kv_cache_offset_not_mutated_in_place | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_attention_caches_batch_select_and_merge; upstream-specific empty/rotating/offset cases MISSING; batching excluded |
| test_batch_rotating_kv_cache | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_attention_caches_batch_select_and_merge; upstream-specific empty/rotating/offset cases MISSING; batching excluded |
| test_cache_copying | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_kv_cache_grows_in_capacity_chunks_and_snapshots_active_state; no exact upstream copy contract claimed |
| test_cache_list | PASS | MISSING generic CacheList composition equivalent; no alpha claim |
| test_cache_to_quantized | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_quantized_kv_cache_uses_capacity_growth_for_each_component; no equivalent model-conversion assertion; quantized KV excluded |
| test_cache_with_generate | PASS | SEMANTIC OVERLAP: tests/models/test_alpha_cache_semantics.py::test_fp16_kv_representation_append_reset_and_request_isolation; actual generation/continuation in stability/*/*/result.json; frozen full/cache elementwise gate remains separate |
| test_extend_with_empty_and_nonempty_batch_caches | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_attention_caches_batch_select_and_merge; upstream-specific empty/rotating/offset cases MISSING; batching excluded |
| test_extend_with_empty_batch_cache_preserves_dtype | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_attention_caches_batch_select_and_merge; upstream-specific empty/rotating/offset cases MISSING; batching excluded |
| test_make_mask_with_cache | PASS | SEMANTIC OVERLAP: tests/models/test_alpha_cache_semantics.py::test_exact_window_boundaries_keep_absolute_positions_and_chunk_visibility and test_analytical_projection_rotary_positions_visibility_and_reset; different cache storage implementation |
| test_merge_with_empty_caches | PASS | PARTIAL: tests/test_core_cache_capacity.py::test_attention_caches_batch_select_and_merge; upstream-specific empty/rotating/offset cases MISSING; batching excluded |
| test_rotating_cache_mask | PASS | SEMANTIC OVERLAP: tests/models/test_alpha_cache_semantics.py::test_exact_window_boundaries_keep_absolute_positions_and_chunk_visibility and test_analytical_projection_rotary_positions_visibility_and_reset; different cache storage implementation |
| test_rotating_cache_updates | PASS | SEMANTIC OVERLAP: tests/models/test_alpha_cache_semantics.py::test_exact_window_boundaries_keep_absolute_positions_and_chunk_visibility and test_analytical_projection_rotary_positions_visibility_and_reset; different cache storage implementation |
| test_save_load | PASS | MISSING exact serialization round trip; persistence is outside alpha scope |
| test_save_load_arrays_cache | PASS | MISSING exact serialization round trip; persistence is outside alpha scope |
| test_save_load_batch_caches | PASS | MISSING exact serialization round trip; persistence is outside alpha scope |
| test_save_load_cache_list | PASS | MISSING exact serialization round trip; persistence is outside alpha scope |
| test_save_load_mixed_cache | PASS | MISSING exact serialization round trip; persistence is outside alpha scope |
| test_save_load_quantized_cache | PASS | MISSING exact serialization round trip; persistence is outside alpha scope |
| test_save_load_rotating_cache | PASS | MISSING exact serialization round trip; persistence is outside alpha scope |
| test_trim_cache | PASS | PARTIAL: tests/test_cache_runtime.py::test_block_kv_growth_materialization_clone_and_trim; block mode excluded; exact dense trim/generation case MISSING |
| test_trim_cache_with_generate | PASS | PARTIAL: tests/test_cache_runtime.py::test_block_kv_growth_materialization_clone_and_trim; block mode excluded; exact dense trim/generation case MISSING |
| test_window_mask_with_full_kv_cache | PASS | SEMANTIC OVERLAP: tests/models/test_alpha_cache_semantics.py::test_exact_window_boundaries_keep_absolute_positions_and_chunk_visibility and test_analytical_projection_rotary_positions_visibility_and_reset; different cache storage implementation |

[final-checks/upstream-original-with-asset.json](../archived-evidence.md).
