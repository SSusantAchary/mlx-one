# MLXBoost Phase 0 — current acceptance status

The missing-toolchain blocker is resolved: Xcode 26.3 (17C529), Metal 32023.864,
macOS SDK 26.2, macOS 15.8.2 / Apple M4 32 GiB. Both `metal` and `metallib` resolve.

- Native source tests: **123 passed, zero failed/skipped**.
- The same tests against the independently installed native wheel: **123 passed**.
- Full regression: **2,391 passed, 84 skipped, 27 pre-existing failures**;
  every failure identity matches the preserved baseline.
- Numerical matrix: **180 cases**, FP16/FP32, maximum absolute error **0.0**.
- Native and standard wheels separately installed in fresh venvs, imported and
  executed outside the repository. The post-native-build standard wheel has no binaries.
- Two paired benchmark runs: **1,200 synchronized samples**. Offline tuning compares
  64/128/256 for both dtypes; all proposals remain unapproved and EXPERIMENTAL.
- Metal capture generated (135,873,287 bytes); Xcode GUI readability not inspected.
- Phase 0 reference model profiling now passes on pinned Qwen2.5-0.5B Q8: 128
  input tokens / 16 requested output tokens, cold request, one warmup and three
  measured requests. This uses ordinary MLX-One reference math and no custom model kernel.
  [Model record](mlxboost-native-phase0-results/native-validated/model-reference/).
  Phase 0 acceptance evidence is frozen before Phase 1-G03 implementation.

Current native wheel: `local/mlxboost-phase0/resume/dist-native-final/mlx_one-0.1.2-cp312-cp312-macosx_15_0_arm64.whl`
(SHA256 `d20628515f8fefd436b508b1815136c3b7afbbe21dc3e725ece9b8c18f4661fe`).
Current standard wheel: `local/mlxboost-phase0/resume/dist-standard-final/mlx_one-0.1.2-py3-none-any.whl`
(SHA256 `b9a150436d56d057b5d91df29b7fafe27891d03f6fcc29e7bd2b251ea8154103`).

The native wheel contains `_ext.cpython-312-darwin.so`, `libmlxboost_ops.dylib`,
`mlxboost.metallib`, `_build_info.json`. Installed Mach-O objects are arm64 with
@loader_path rpaths and no developer-specific library dependencies.

[Current machine-readable evidence](mlxboost-native-phase0-results/native-validated/validation-summary.json),
[wheel receipts and archive inventory](mlxboost-native-phase0-results/native-validated/),
[historical initial report](mlxboost-native-phase0-results/pre-toolchain-report.md).
Raw build/test logs are under `local/mlxboost-phase0/resume/logs/`.

Native kernels and timings remain demonstration-only. Raw native axpby measurements
exclude repeated integrity checking: opted-in unqualified guarded selection measured
16.286 ms median, while default-off selection measured 655 ns/call. No model
inference acceleration or runtime approval is claimed. Phase 1-G03 quantized GEMV
is a separate user-authorized milestone with independent gates.
