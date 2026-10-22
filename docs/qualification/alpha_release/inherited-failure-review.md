> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Inherited failure review

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Entry entire native suite: {'passed': 2066, 'failed': 8, 'skipped': 18}. Final default: {'passed': 1834, 'failed': 0, 'skipped': 67}. Final native: {'passed': 2099, 'failed': 0, 'skipped': 18}. All eight entry failures were reproduced and scoped repairs made: patch both the parent mlx package and mlx.core in fake fixtures; use supported group 64 and a valid-width QLoRA fixture; check backend-free CLI in a fresh process; compare the actual 29-registration inventory. HTTP terminal flush and packed-weight expectations remain intact. No XFAIL or threshold relaxation was added. All eight dispositions are FIXED when the final native suite is green; any new failure remains blocking.

[baseline-native.log](../archived-evidence.md); [inherited-repair.log](../archived-evidence.md); [trained-controls/regression-native-final.log](../archived-evidence.md); [inherited-failure-review.json](../archived-evidence.md). Ruff’s 65 inherited findings remain in [LINT-001](lint-debt.md), with no newly introduced findings; waiver/disposition is pending.


| Exact baseline test | Original failure | Correction | Final disposition |
|---|---|---|---|
| `tests/test_engine_observation.py::test_terminal_text_flush_reaches_http_without_adding_a_token` | HTTP SSE error event: Cannot interpret mlx.core.float16 as a NumPy dtype | Fixture isolation: patch both mlx and mlx.core so NumPy never calls real Metal dtype conversion; retain text-flush/token-count assertions | FIXED; native JUnit PASS |
| `tests/test_native_tuning_mlx.py::test_native_adapter_round_trip_and_trainable_boundary[qlora]` | mx.quantize rejects unsupported group_size=8 before round-trip assertions | Fixture config: hidden width 64, intermediate width 128 and supported group_size=64; preserve trainable boundary and saved adapter round trip | FIXED; native JUnit PASS |
| `tests/test_p0_engine_qualification.py::test_fp16_conversion_preserves_native_affine_payload[4]` | Fake floating tensor astype receives real mlx.core.float16 | Fixture isolation: patch the parent mlx package and mlx.core together; preserve exact packed payload and floating conversion assertions | FIXED; native JUnit PASS |
| `tests/test_p0_engine_qualification.py::test_fp16_conversion_preserves_native_affine_payload[8]` | Fake floating tensor astype receives real mlx.core.float16 | Fixture isolation: same repair, separate Q8 payload cell | FIXED; native JUnit PASS |
| `tests/test_p0_engine_qualification.py::test_explicit_four_bit_projection_uses_existing_rowwise_primitive` | Native gather_qmm receives NumPy arrays from a partial fake module | Fixture isolation: parent mlx.core now points at matching fake core; preserve existing rowwise primitive invocation assertion | FIXED; native JUnit PASS |
| `tests/test_p1_runtime.py::test_explicit_float_conversion_preserves_packed_weights` | Fake floating tensor astype receives real mlx.core.float16 | Fixture isolation: patch both mlx and mlx.core; packed weight/scales/bias preservation still asserted | FIXED; native JUnit PASS |
| `tests/test_p1_runtime.py::test_p1_profile_cli_is_backend_free_and_keeps_native_pending` | Backend-free assertion reads mlx.core imported by earlier native tests | Isolation: execute unchanged CLI contract in a fresh subprocess and assert it never imports mlx/MLX core | FIXED; native JUnit PASS |
| `tests/test_task_qualification_native.py::test_isolated_native_task_worker_and_pending_matrix` | Registry count assertion expects 27 but actual inventory is 29 | Stale expectation: require 29 and equality with actual complete architecture inventory; no registration removed | FIXED; native JUnit PASS |

The interim newly added GPT-2 analytical test failure was a fixture mismatch (n_ctx and BOS/EOS outside the tiny vocabulary). Its failed run is retained in final-checks/regression-native.log; the corrected 13-case analytical group and final entire native suite pass. No result was deleted.
