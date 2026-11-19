# MLXBoost native Phase 0 audit

Initial audit date: 15 November 2026. Starting commit: `4cad1634562c5c12933882deb0b8ab3c9461dd97`,
branch `feat/gptoss-git-upgrade`; implementation branch `feat/mlxboost-native-phase0`.
The original `git status --porcelain=v1` was saved to `/tmp/mlxboost-phase0-initial-status.txt`.
README.md and mlx-one.png were modified; numerous benchmark, documentation, logo,
and qualification files were untracked. All are outside this implementation's edits.

## Baseline and reuse

The local package is **0.1.2**, not the prompt's public 0.1.4. Setuptools uses
`src/`, with MLX and mlx-metal pinned to 0.32.2. There was no setup.py or native
MLXBoost extension. MANIFEST.in explicitly prunes benchmarks/tests/scripts/docs.
The named development guidelines file is absent in this checkout; the sibling
`mlx-one-release-014` copy was read as context, not imported as repository policy.
No applicable AGENTS.md was found.

| Feature | Already implemented / reuse | Proposed change |
| --- | --- | --- |
| F01 | Backend-free public imports; opt-in `mlxboost/config.py`; no custom Phase 0 native primitive | Conditional setup.py, CMake/nanobind primitive, Metal library, lazy native namespace |
| F02 | `engine/kernel_dispatch.py` contains shape/precision checks for stock MLX ops; `engine/selection.py` requires exact checksummed runtime evidence | Separate demo-only immutable request/decision and exact kernel evidence; leave engine policies untouched |
| F03 | `ops/profiling.py` synchronizes before/after observed phases; `engine/timing.py` distinguishes model and SSE windows; `diagnostics.py` probes hardware explicitly | Callable profiler with cold/warm/raw synchronized samples, provenance, JSON/Markdown, optional capture |
| F04 | Runtime qualification registry distinguishes experimental/correctness/performance; no bounded native-kernel offline tuner | Bounded axpby candidates, immutable content-addressed records, separate reviewed read-only manifest |

Existing `Engine`, `InferenceActor`, `BatchScheduler`, `CacheManager`, reference_dense,
fast_decode, GPT-OSS experiments, paged/hybrid caches and cold prefix storage retain
their current ownership. Failed Qwen and cache-sensitive numerical cases stay failed.
No model module, model equation, checkpoint, inference route or support catalog is
in scope. Native demo evidence never qualifies a model.

## Environment and baseline checks

Host arm64; system Python 3.14.7; development interpreter `.venv/bin/python`,
Python 3.12.11; mlx=mlx-metal=0.32.2; setuptools 84.0.0, build 1.6.0,
pytest 9.1.1; CMake 4.4.5. nanobind was absent (installed for this work).
Active developer directory is CommandLineTools; xcodebuild requires full Xcode;
`xcrun --find metal` fails. No full Xcode found in /Applications.
The sandbox cannot load a Metal device; an authorized outside-sandbox probe
reports Apple M4, applegpu_g16g, 34,359,738,368 bytes unified memory, Metal available.
Toolchain absence and sandbox GPU access are separate limitations.

Baseline command: `.venv/bin/python -m pytest tests/mlxboost
tests/optimization/test_math_policy.py tests/alpha_0_1/test_package_imports.py
tests/alpha_0_1/test_packaging_contract.py -q --no-cov`:
**78 passed, 10 skipped** (explicit GPU/checkpoint opt-in tests).

## Source authority and licensing

MLX v0.32.2 extension.py, example setup/CMake/bindings/axpby sources and primitive
headers were fetched and inspected. Use CMakeExtension/CMakeBuild, NB_DOMAIN mlx,
MLX streams/encoders and allocator ownership. Do not adopt the upstream example's
transform code: the demo will explicitly reject differentiation/vectorization.
ZMLX revision `d8cd0d88d4299ca6821d706f9d5b4a3520d688f5` patch selection,
registry, profile, metal wrapper, autotune and cache were inspected. Reuse ideas
(exact eligibility, synchronized timers, bounded configurations), not their runtime
autotune decorator or fallback timing values. Both projects have MIT licenses;
retain Apple's notice for extension API adaptations. No ZMLX runtime dependency.

## Planned files

Add `src/mlx_one/mlxboost/native/` (Python records, loader, dispatcher, registry,
profiler, offline tuner, CLI, C++/Metal `_build/`, source notices). Add setup.py;
extend only package-data and MANIFEST.in for native sources/assets. Add
`tests/native/`, `benchmarks/native/`, build/validation scripts, and Phase 0
documentation/evidence. Preserve the README and existing MLXBoost code.

Acceptance: default wheel free of binaries even after native builds; explicit
runtime opt-in plus exact approved evidence; malformed/stale data fails closed;
strict typed failures; real GPU parity/benchmarks only when execution succeeds;
no invented results. Native build was initially blocked until a Metal compiler existed.

## Toolchain continuation

The user installed Xcode 26.3 (17C529) and the Metal Toolchain. Both `xcrun --find
metal` and `xcrun --find metallib` now resolve. The build, native wheel, 123 Phase 0
tests and 180 numerical cases pass on M4. The resumed implementation fixed MLX's
interaction with an unrelated DEBUG=release environment value, converted MLX
SmallVector strides explicitly in nanobind, and replaced process-substitution
logging in the scripts. Generated editable binaries are excluded from Git and
from the ordinary wheel. See the current [evidence report](mlxboost-native-phase0-evidence.md)
for final builds, immutable measurements, actual tuning and unchanged baseline failures.
