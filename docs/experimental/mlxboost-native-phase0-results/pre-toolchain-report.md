# MLXBoost Phase 0 implementation evidence

15 November 2026. **EXPERIMENTAL, mandatory native acceptance incomplete.**
The native build is blocked by the missing Apple Metal compiler. There is no
custom-kernel numerical result, native-wheel validation, tuned winner, approved
runtime manifest or model inference acceleration claim.

## Working tree and environment

Starting commit `4cad1634562c5c12933882deb0b8ab3c9461dd97`, original branch
`feat/gptoss-git-upgrade`, implementation branch `feat/mlxboost-native-phase0`.
Original README/logo changes and untracked user files were preserved. README.md
and mlx-one.png were byte-compared against the preserved baseline copy after work.
No model, engine, HTTP/SSE, cache or existing MLXBoost source was edited.

Local package version is **0.1.2**. Development interpreter is
`<repo>/.venv/bin/python`, Python 3.12.11; the system python3 is 3.14.7.
mlx=mlx-metal=0.32.2; setuptools 84.0.0, build 1.6.0, nanobind 2.15.0,
CMake 4.4.5. Device: Apple M4, applegpu_g16g, 34,359,738,368 bytes unified
memory. Exact macOS/compiler/SDK/load details are in the raw measurement JSON.
The active developer directory is CommandLineTools. xcodebuild requires full
Xcode; `xcrun --find metal` fails. No full Xcode was found in /Applications.
MLX cannot access Metal inside this sandbox; explicit GPU tests/measurements ran
outside it with authorized tool execution. This does not resolve the missing compiler.

See [audit](../mlxboost-native-phase0-audit.md) and
[upstream revisions, file hashes and licenses](../mlxboost-native-phase0-sources.json).
MLX v0.32.2 is build authority. ZMLX source study is pinned to
`d8cd0d88d4299ca6821d706f9d5b4a3520d688f5`; there is no ZMLX runtime dependency.

## Implemented source and checked boundaries

| Ticket | Implementation | Observed validation |
| --- | --- | --- |
| F01 | Conditional setup.py; pinned CMake/nanobind bridge; C++ inference primitive; standalone Metal kernel; relocatable library design; pre-dlopen ABI/source guard; lazy status | CMake config and both C++ translation units compile; standard wheel passes. Metal/native wheel/GPU primitive unverified |
| F02 | Immutable request/candidate/config/decision; exact capability/evidence dispatch; default OFF; strict typed failures; no engine integration | Positive synthetic fixture, negative manifests/scopes/ABI/source checks, concurrency and actual absent-extension fallback pass |
| F03 | Evaluated/synchronized timing, randomized order, raw percentiles/memory/provenance, JSON/Markdown; optional capture; local reference model harness | Timer completion tests pass; 600 real reference GPU samples, import/dispatch measurements; model/capture unavailable |
| F04 | Offline 64/128/256 search, real pipeline resource limit, correctness before timing, budgets, deterministic ranking; immutable runs and unapproved proposals | Schema/integrity/ranking/budget tests pass; actual tuning cannot execute without extension |

New sources live under `src/mlx_one/mlxboost/native/`, with focused tests under
`tests/native/`, developer harnesses under `benchmarks/native/`, and two build/
validation scripts. Existing metadata/pins are retained; only native package data
and sdist-source inclusion were added to pyproject.toml/MANIFEST.in.

## Commands actually executed

All relative paths below start at the repository root. Tool invocations and raw
logs are retained under `local/mlxboost-phase0/logs/`. Temporary baseline copies
were created under /tmp; the original working tree was never reset.

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git switch -c feat/mlxboost-native-phase0
python3 -V
xcodebuild -version
xcrun --find metal
cmake --version
.venv/bin/python -m pip install 'nanobind>=2.4,<3'

# Initial relevant baseline: 78 passed, 10 skipped.
.venv/bin/python -m pytest tests/mlxboost tests/optimization/test_math_policy.py \
  tests/alpha_0_1/test_package_imports.py tests/alpha_0_1/test_packaging_contract.py -q --no-cov

# Native build: fails with an intelligible missing-Metal-compiler error.
MLX_ONE_BUILD_NATIVE=1 .venv/bin/python -m build --wheel --no-isolation \
  --outdir local/mlxboost-phase0/dist-native

# Configure succeeds; the subsequent actual metallib target fails, xcrun error 72.
cmake -S src/mlx_one/mlxboost/native/_build -B local/mlxboost-phase0/cmake \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/local/mlxboost-phase0/compile-check" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build local/mlxboost-phase0/cmake --parallel 2

# Both axpby.cpp and bindings.cpp commands from compile_commands.json were
# separately executed successfully (C++20, actual pinned MLX headers).
# This is an object compilation check, not a linked native extension test.

MLX_ONE_BUILD_NATIVE=0 .venv/bin/python -m build --wheel \
  --outdir local/mlxboost-phase0/dist-standard-verified
.venv/bin/python -m build --sdist --no-isolation --outdir local/mlxboost-phase0/dist-source

# Full final suite with GPU access; explicit Phase 0 GPU availability checks.
MLXBOOST_PHASE0_GPU_TESTS=1 MLX_ONE_NATIVE=0 .venv/bin/python -m pytest tests \
  -q --no-cov --junitxml=/tmp/mlxboost-phase0-final-tests.xml

.venv/bin/python benchmarks/native/bench_phase0.py \
  --output local/mlxboost-phase0/final-measurements/run1 --trials 50 --warmup 10
.venv/bin/python benchmarks/native/bench_phase0.py \
  --output local/mlxboost-phase0/final-measurements/run2 --trials 50 --warmup 10
.venv/bin/python benchmarks/native/bench_overhead.py --output local/mlxboost-phase0/final-overhead
.venv/bin/python -m mlx_one.mlxboost.native tune --output local/mlxboost-phase0/tuning-final
# Last command exits 2: tuning unavailable: extension_unavailable.
```

The first full suite inside the sandbox aborted during an existing MLX backend
import. It was rerun with actual GPU access; no successful result is attributed
to the aborted run. A baseline archive at the starting commit, overlaid with the
original user changes and the same retained local assets, was also tested outside
the sandbox. Git metadata and source captures were restored in that archive so
path-sensitive evidence tests ran in the same scope.

## Test results

| Group | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Initial affected baseline | 78 | 0 | 10 |
| Full preserved baseline | 2,268 | 27 | 84 |
| Full changed tree, explicit Phase 0 GPU session | 2,333 | 27 | 141 |
| Phase 0 alone, explicit GPU session | 65 | 0 | 57 |
| Combined focused suite without GPU opt-in | 135 | 0 | 75 |

Full baseline duration: 43.48 s; changed-tree duration: 43.11 s. Those single suite
wall times are not a performance regression measurement. Both suites also report
nine passing subtests and two warnings. Every failing test identity matches the
preserved baseline: **zero new failures**. The 26 asset failures concern absent
DistilBERT/NER/CLIP/depth/DETR/ViT checkpoint/reference files; the remaining failure
is the original README performance-summary test. These were not hidden or relabeled.
Machine-readable groups and failure-ID digests:
[test-results.json](test-results.json).

The 57 new GPU skips require a compiled extension: 54 dtype/shape/threadgroup
cases, transform/error checks, repeated-execution checks, and the tuning test.
Eight new actual GPU fallback cases run successfully without that extension.
Existing asset/GPU opt-ins retain their original skips. Applicable existing
API/CLI/HTTP/SSE contract and tiny-model tests are in the full suite; this is not
a real-checkpoint ten-model serving campaign.

Scoped ruff checks pass; shell syntax checks pass; git diff --check passes.
Backend-free imports and status inspection pass in subprocesses forbidding MLX,
Torch, Transformers and mlx-lm imports. No native initialization occurs on import.

## Independent wheel and sdist evidence

Verified standard wheel:
`local/mlxboost-phase0/dist-standard-verified/mlx_one-0.1.2-py3-none-any.whl`

SHA256: `e90f06d11ac74aa2e765972d38f0a683b582533e44cb3dfd670b3c4ebef9d257`.
It has 525 entries, including 16 native Python/source/notice entries, and **no**
`.so`, `.dylib`, `.metallib` or generated build metadata. Benchmarks/tests/local
results are excluded. The sdist contains the C++/Metal/CMake/build-identity sources.
[Archive inventory](standard-wheel-archive.json).

A fresh virtualenv installed the ordinary wheel and its dependencies without
system-site-packages or source-path injection. The final verified wheel was then
reinstalled in that isolated environment and tested from /private/tmp:

```bash
.venv/bin/python -m venv local/mlxboost-phase0/venv-standard
local/mlxboost-phase0/venv-standard/bin/python -m pip install \
  local/mlxboost-phase0/dist-standard/mlx_one-0.1.2-py3-none-any.whl
local/mlxboost-phase0/venv-standard/bin/python -m pip install --no-deps --force-reinstall \
  local/mlxboost-phase0/dist-standard-verified/mlx_one-0.1.2-py3-none-any.whl
# After cd /private/tmp, using absolute paths:
<repo>/local/mlxboost-phase0/venv-standard/bin/python <repo>/benchmarks/native/wheel_smoke.py \
  --flavor standard --output <repo>/local/mlxboost-phase0/standard-wheel-smoke-verified.json
```

Installed-package ownership, public imports, lazy backend behavior, real GPU
reference output, native opt-in fallback and strict missing-extension exception
all pass. Installed CLI reports `mlx-one, version 0.1.2`.
[Smoke receipt](standard-wheel-smoke.json).

**Native wheel: not built.** No filename/platform tag, linked module/dylib,
metallib, rpath audit or fresh native-wheel GPU smoke can be supplied honestly.
The native build's exact blocker is `xcrun: unable to find utility "metal"`.
The C++ object compilation checks do not waive any of these gates.

## Numerical and fallback evidence

Actual GPU fallback matrix: FP16/FP32, lengths 0/1/65/4096, coefficients .75/-.25,
seed 7, native request OFF/ON. All 16 cases select mlx_reference, with max absolute
error **0.0** against ordinary MLX arithmetic. Empty comparisons are vacuous;
their recorded error is 0.0 by convention. Asserted tolerance is atol=rtol=1e-3.
Strict native mode raises extension_unavailable. This demonstrates fallback,
**not native kernel parity**.
[Matrix and environment](fallback-matrix.json).

| Scenario | Observed evidence |
| --- | --- |
| Native flag OFF | Real reference tensor output, native_disabled; pure dispatcher test forbids backend probes |
| Missing extension with opt-in | Real standard-wheel/reference output, extension_unavailable |
| Strict opt-in | Actual NativeKernelUnavailable(extension_unavailable) |
| Unapproved/stale/corrupt/mismatched evidence | Backend-free fixture tests reject approval=false/1, changed source/binary, changed scope, missing/corrupt proof, invalid timings, traversal/symlinks, duplicate JSON/NaN |
| ABI mismatch | Loader fixture tests prove mismatched installed/compiled version or libmlx hash is rejected before backend import/dlopen |
| Qualified match | Synthetic test fixture only; no real qualification approved |

Custom primitive parity, FP16 rounding, NaN/Inf, zero-size native allocation,
streams/lifetimes and native transform rejection remain **unverified** until the
Metal extension can run. Their test implementations are present and skipped.

## Actual profiler and tuner measurements

Below are **reference axpby synchronized wall times**, milliseconds, including
Python construction, MLX evaluation and GPU completion. They are not GPU-only
kernel timings and not model inference measurements.

| Dtype | Length | Run 1 median | Run 2 median |
| --- | ---: | ---: | ---: |
| FP16 | 256 | 0.216251 | 0.224479 |
| FP16 | 4,096 | 0.224021 | 0.197521 |
| FP16 | 65,536 | 0.247625 | 0.175709 |
| FP32 | 256 | 0.218209 | 0.177771 |
| FP32 | 4,096 | 0.223062 | 0.169271 |
| FP32 | 65,536 | 0.255396 | 0.184125 |

Each of 12 cases has 50 measured samples, 10 separate warmups and a separately
recorded first profiled call: **600 measured samples** total. Raw JSON contains
all samples, p90/p99, dispersion, MLX/process memory and exact provenance.
Matching Markdown summaries are alongside the JSON under
[run1](run1/) and
[run2](run2/). Every native comparison is explicitly
skipped. Thermal state, driver caches and unrelated host load were uncontrolled;
the run-to-run variation is not attributed to code or a speedup.

Ten fresh subprocesses measured an incremental optional-namespace import median
of **2.1213545 ms**, after the existing public namespace import. Thirty batches of
10,000 default-off selections measured **626.62085 ns/call** median. The latter is
a batch-derived per-call wall time; ordinary model routes never call the new demo
dispatcher. Raw [host overhead](host-overhead.json)
also records distinct RSS observations; these numbers do not prove model-level
baseline performance neutrality.

Tuner configurations implemented: **64/128/256** threads, correctness first,
bounded memory/time. **No candidate measurement, ranking winner or promotion was
produced**, because the extension is absent. GPU capture is unavailable without
full Xcode. Existing Qwen2.5-0.5B Q8 and Qwen3-0.6B Q8 assets were not present;
the retained local checkpoint-path record also points to an absent snapshot.
No downloads, replacement models, GGUF runtime substitution, 4k/32k runs or
real-checkpoint HTTP/SSE acceleration measurements were attempted.

## Readiness and rollback

F01–F04 source and tests are delivered; mandatory native validation is blocked.
Keep **EXPERIMENTAL**. No model/support status or numerical policy was promoted.
To finish acceptance, activate a Metal-capable Xcode/toolchain so `xcrun --find metal`
succeeds, then run the provided build and validation scripts, install the native
wheel in another fresh environment, run its smoke/parity matrix and retain real
tuning results. Full linked-library relocation must be checked then.

Disable routing with MLX_ONE_NATIVE=0; a fresh environment using the verified
standard wheel is the rollback. Revert only the new native namespace and focused
build/package-data/tests/scripts/docs additions if source rollback is desired.
Preserve the original README/logo/untracked work. Production model kernels and
model-path integration are future work requiring separate numerical and
end-to-end performance evidence.
