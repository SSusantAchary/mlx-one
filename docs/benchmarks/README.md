# Model performance methodology

The [README table](../../README.md#model-performance) is generated from public JSON evidence,
not example numbers. [Inventory](inventory.md), [machine-readable inventory](inventory.json),
and [coverage](coverage.md) track all 13 pinned text inference checkpoints in the current
support manifest. Nominal parameter sizes describe the advertised model class (MoE total /
active), not the number of packed integers or a count calculated from weight bytes.
Other task, training and discovery candidates in `docs/model-catalog.md` are outside this
text inference comparison.

## Historical evidence: 128 input / 16 output

`historical/*-performance.json` are byte-for-byte copies of all 16 performance records from
`docs/qualification/alpha_release/numerical-correction-20261021-1792560186243989000/final-checks/`.
Repository cleanup moved that original tree to ignored
`local/evidence/repository/docs/qualification/alpha_release/`.
Only the **128-context, production_fp16** row is published. Natural-state and 512/1024-context
samples remain in the raw files and are not combined with it. Each published sample has
128 actual input tokens and 16 actual output token IDs. Nine checkpoints have primary
precision coverage; seven also have separately measured stored-source BF16 coverage.

These are direct calls to native models on the model executor thread, **not Python Engine
requests**, despite using ModelManager to load and own the model. No HTTP, CLI, queueing,
chat-template or scheduler latency is included. The backend projects full sequence logits.
Affine Q4/Q8 conversion uses group size 64 and FP16 floating execution; stored source
precision is preserved. Attention KV is explicitly FP16; convolution/sliding state retains
its implementation's layout. Batch size is one; fresh state prevents prefix reuse.

The host was Apple M4 (Mac16,12 MacBook Air), 32 GB, macOS 15.7.9, Python 3.12.11,
MLX/Metal 0.32.2. Each shape/state mode has one discarded warmup followed by three measured
requests. Each table cell is an independent median; it need not describe the same repetition
as another cell. In particular, median prefill speed and median TTFT can be reciprocal for
this cohort, while Engine timing from other reports measures different intervals.

TTFT times the full prefill forward and evaluated first-token argmax. Prefill speed is actual
input tokens divided by this duration. Decode speed counts **actual generated tokens minus
one** divided by the subsequent decode duration: the first token belongs to prefill. The
archived runner fixes a budget of 16 and checks EOS after subsequent decode steps. The
published 16-output rows are unaffected by its first-token EOS limitation. Tokenization,
load/conversion and warmup are outside timing. EOS token IDs count as actual outputs.

MLX peak memory resets after warmup and before each inference request. It includes resident
weights and MLX allocations during inference; it excludes the earlier load/conversion peak.
It is not process RSS, wired memory, or KV-only memory. GiB means bytes / 1024³. Full token
IDs and state memory remain in the JSON. Rounding happens only while rendering Markdown.

`historical/cohort.json` records the source environment, immutable revisions through the
records, original source asset hashes, production source hashes, original command receipt
hashes, and SHA-256 of each unchanged performance report. `historical/runner.py.txt` is the
exact archived driver retained as evidence, not an instruction to run old qualification.
The original worktree was dirty; the entry commit alone does not identify its implementation.
The individual source hashes preserve that distinction. Results are historical experimental
measurements, not release approval, numerical certification or comparisons against another
framework.

To re-import preserved data (read-only with respect to the archive):

```sh
python benchmarks/model_performance/import_historical.py \
  local/evidence/repository/docs/qualification/alpha_release/numerical-correction-20261021-1792560186243989000
```

This copies measurements only. It does not execute or reinstate numerical qualification.

## Standard protocol: 128 input / 128 output budget

`runs/M01-direct-128-128.json` is a fresh end-to-end runner validation on the already cached
GPT-2 snapshot on macOS 15.8.1: three repetitions each produced 128 actual outputs.
`runs/M01-runner.py.txt` preserves the exact measured runner revision. It is published in its
own standard cohort, separately from the 16 historical configurations. No large checkpoints
were downloaded or loaded. The other 12 primary configurations await standardized runs.


`benchmarks/model_performance/run.py` executes **direct-model native MLX on Metal**, with
last-token-only logits, batch size one, greedy sampling, FP16 KV and fresh caches. It runs
one full warmup request and three measured requests by default. This differs from the
historical full-logits backend and must remain a separate cohort. The runner never downloads
assets and accepts only existing pinned Hugging Face cache snapshots with matching config
hashes. It hashes actual weight shards before timing and checks archived weight hashes where
available. For newly planned weights, the cache repository/revision layout plus pinned config
identifies origin; the freshly computed shard hashes identify the exact bytes measured.
They are not an independent remote attestation of those new shard contents.

The default prompt repeats the historical seed text and slices to exactly 128 token IDs.
The IDs and their hash are recorded; optionally supply `--token-ids prompt.json` containing
exactly 128 valid IDs for the checkpoint tokenizer. No chat template is applied. EOS is
respected, including first-token EOS. Requested output budget is never substituted for actual
outputs. Early-EOS results are retained and labeled with their actual output count in a
separate cohort; they do not fill standardized **128 actual output** coverage. With no decode
steps the decode throughput is undefined (shown as a dash).

Each forward evaluates argmax and synchronizes Metal before timing boundaries. TTFT covers
prefill through the first token; decode speed is `(actual_outputs - 1) / decode_seconds`.
Allocation peaks reset per request after loading/conversion and warmup. No sampling, layer
profiling, tokenizer download, or reference model is active during measurement. The record
contains actual host/device, runtime versions, prompt and output IDs, finish reasons, durations,
weight/config hashes, runner hash and production source hashes. Use a fresh process per model
and precision; keep the machine idle and run one GPU workload at a time. The preflight weight
size and MLX memory limit are conservative safeguards, not a promise that a model fits.

The generation benchmark follows the [functional admission policy](../model-admission.md).
Benchmark speed does not promote checkpoint status or establish answer quality. Gemma3 and
Qwen3.5 limitations remain in their model guides and the inventory. Do not run retired
exhaustive numerical gates to admit or benchmark a checkpoint.

## Manual runs after download

Activate the repository environment. Download the exact revisions yourself using your normal
Hugging Face workflow; no commands below download weights. Set each path to its already
populated cache snapshot (the `models--ORG--NAME/snapshots/REVISION` layout is required).
Use `HF_HUB_CACHE` if your cache lives elsewhere.

```sh
source .venv/bin/activate
BENCH_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

python benchmarks/model_performance/run.py \
  --model-id LiquidAI/LFM2.5-24B-A2B-MLX-4bit \
  --checkpoint "$BENCH_HUB_CACHE/models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/667408d494bde191c2cb9dd4f4549827816b9275" \
  --output docs/benchmarks/runs/M11-direct-128-128.json

python benchmarks/model_performance/run.py \
  --model-id mlx-community/gpt-oss-20b-MXFP4-Q4 \
  --checkpoint "$BENCH_HUB_CACHE/models--mlx-community--gpt-oss-20b-MXFP4-Q4/snapshots/f356f2747216d7e98fee755df25987459fc19089" \
  --output docs/benchmarks/runs/M12-direct-128-128.json

python benchmarks/model_performance/run.py \
  --model-id manjunathshiva/gpt-oss-20b-tq3 \
  --checkpoint "$BENCH_HUB_CACHE/models--manjunathshiva--gpt-oss-20b-tq3/snapshots/20ba1835e625264e07366d215933fe31a9940a6a" \
  --output docs/benchmarks/runs/M13-direct-128-128.json

python benchmarks/model_performance/report.py --write
python benchmarks/model_performance/report.py --check
```

All three checkpoints use their mixed stored precision; do not pass `--profile affine_q4`
to requantize them. GPT-OSS may spend the budget in analysis before final text; the benchmark
counts raw token IDs regardless of rendered channel. For the small source checkpoints,
select `--profile affine_q4` or `--profile affine_q8` as listed in `inventory.json` to benchmark
that primary conversion. Output files cannot be overwritten. To compare repeated campaigns,
retain their raw records outside `runs/` and curate one record per exact cohort/model/profile
for publication; duplicate comparable runs deliberately fail validation.

## Python Engine, CLI and HTTP measurements

The direct table makes no claims about these interfaces. Existing interface runners remain
separate: `benchmarks/model_parity/reference_performance.py` measures Python Engine requests,
while `benchmarks/turboquant/interfaces.py` exercises CLI/HTTP paths. See the
[TQ3 manual validation guide](../implementation/turboquant-tq3/manual-validation.md) for their
commands and metrics. Their throughput/TTFT intervals, tokenization, streaming and queue
costs differ; never import those results as direct-model rows. The table generator explicitly
rejects non-direct records. A future serving comparison needs its own protocol, evidence,
actual token accounting and labeled table.

## Regeneration and checks

```sh
python benchmarks/model_performance/report.py --write
python benchmarks/model_performance/report.py --check
python -m unittest discover -s tests/unit -p test_model_performance_reporting.py
```

`--write` updates only the generated README block and inventory/coverage Markdown/JSON.
`--check` verifies immutable record hashes, pinned config/revisions, interface, actual token
counts, durations where recorded, finite metrics, repetitions, cohort boundaries and generated
file freshness without importing MLX or loading a model. Cohorts also separate runner /
production source fingerprints, prompt recipes, warmup counts and actual output counts. Historical raw durations were not
recorded; their reported speeds are preserved, with accounting checked against the archived
runner and actual output IDs, rather than reconstructed from the requested budget. The CI
workflow performs these checks without model downloads or GPU inference.
