# mlx-one — compact repository summary

Updated: 2026-10-14. Native MLX tooling for Apple Silicon; Python 3.10+, Apache-2.0.

## Repository and runtime

- Native model implementations and shared layers: `src/mlx_one/models/`.
- Python `Engine`, generation, streaming and lifecycle: `src/mlx_one/engine/`.
- HTTP/OpenAI-compatible serving, SSE and cooperative slots: `src/mlx_one/server/`.
- Text, vision, retrieval, audio, training, conversion and evaluation modules live
  under `src/mlx_one/`; support and qualification vary by checkpoint/task.
- Tests: `tests/`. Offline benchmark workers and qualification entry points:
  `benchmarks/`. Reference engines are not inference backends.

P0/P1 are already merged at `b698f88afc1729cd7d2dea53fd5a69f11b60246e`.
Current follow-up branch: `qualification/p1-native-m4`.
The runtime reuses native adapters, `InferenceActor`, `BatchScheduler`, dense/paged/
hybrid runners, transactional cache ownership, prefix indexes, segmented arenas,
memory reservations and checksummed evidence. Optimized modes remain gated;
omitted-mode behavior and public generation/HTTP/SSE interfaces are preserved.
`auto` requires matching qualification evidence, otherwise reference dense.

## Current follow-up and validation

Changes add a staged native qualification driver, identical comparison traffic,
prepared-history decode measurement, internal token-ID capture and stricter
source/precision/workload checks. Repairs cover the parity-writer import defect
and masked/unmasked explicit FP16 attention arithmetic. Floating execution policy
is version 2; older or incomplete policy evidence cannot qualify new execution.

| Check | Observed status |
|---|---|
| Backend-free suite | 773 passed, 24 skipped; two inherited dependency warnings |
| Ruff | 65 inherited findings; zero new findings; changed files pass |
| Native M4 / 32 GiB preflight | GPU, RSS watchdog and MLX memory telemetry passed |
| Qwen3-0.6B INT8 serial diagnostic | Passed, 128-token target prompt and 32 forced steps |
| Dense B1 continuation/state | 33 logits and 1,848 KV comparisons matched exactly |
| Prefix restoration versus full-prompt oracle | **Failed** FP16 tolerance; maximum error 0.064453125 |
| Dense B2/B4, paging, other checkpoints, service/stress/performance | Not run or blocked |

Restored execution matches an independently restored single-request cache exactly,
but both differ from fresh full-prompt execution. Greedy agreement does not waive
the logits gate. Cleanup left zero active handles/reservations. Qwen3-small is the
only downloaded P1 checkpoint; the other five assets remain missing. The driver
enumerates 2,592 optimized legal configurations; their performance runs remain
unexecuted. No native production capability, throughput target or default was promoted.

The next gate is full-prefill versus cached-suffix numerical parity, followed by
dense B2/B4, paging, cross-family correctness, service/lifecycle checks and measured
performance. GPU qualification remains specific to checkpoint, precision, hardware,
shape and workload. Generated evidence stays local under `benchmarks/results/p1/`.

## Push current code, README, summary and architecture image

These commands commit the current task's code/tests/tooling, **this one summary**,
the updated `README.md` and the provided `mlx-one_arch.png` architecture image.
UI edits, other Markdown changes, prompts, raw results and unrelated files remain
local. Previously committed documentation remains in Git history.
No push has been performed by the agent.

1. Inspect the branch and index. Expected branch: `qualification/p1-native-m4`.
   The index is currently empty; if other staged files appear, exclude them before
   continuing so the commit contains only the explicit list below.

   ```sh
   cd /Users/susant-achary/dev_work/git_codes/c_base/mlx-one
   git branch --show-current
   git status --short
   git diff --cached --name-only
   ```

2. Stage only the approved paths. Do not use `git add .`, `git add -A` or `git commit -a`.

   ```sh
   git add -- \
     src/mlx_one/engine/adapters.py \
     src/mlx_one/engine/envelope.py \
     src/mlx_one/engine/qualification.py \
     src/mlx_one/inference_baseline_worker.py \
     src/mlx_one/inference_qualification.py \
     src/mlx_one/models/shared/transformer.py \
     src/mlx_one/native_qualification.py \
     src/mlx_one/p1_benchmark.py \
     src/mlx_one/p1_decode.py \
     src/mlx_one/p1_qualification.py \
     benchmarks/p1_native_qualification.py \
     tests/test_dense_batching.py \
     tests/test_native_qwen_mlx.py \
     tests/test_native_qualification.py \
     tests/test_p1_checkpoint_parity.py \
     tests/test_p1_runtime.py \
     docs/REPOSITORY_SUMMARY.md \
     README.md \
     mlx-one_arch.png
   git diff --cached --name-only
   git diff --cached --stat
   git diff --cached --check
   ```

3. Review those 19 paths, then commit and push the feature branch:

   ```sh
   git diff --cached
   git commit -m "Stabilize native P1 qualification and refresh architecture documentation"
   git show --stat --oneline HEAD
   git push -u origin qualification/p1-native-m4
   git status --short
   ```

Open a pull request from `qualification/p1-native-m4` to `main`. Remaining dirty
files are expected and are not included in this commit or push. Do not force-push.
