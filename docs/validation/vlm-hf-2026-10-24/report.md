# Real VLM image benchmark campaign — 2026-10-24

All five primary checkpoints completed real native Python, CLI and live HTTP/SSE smokes, plus four HF-image benchmark workloads each. These results supersede the earlier source-only pending status for these exact revisions. No commit, push or release was performed.

Hardware: Apple M4, 32 GiB unified memory; native MLX/Metal 0.32.2; Python 3.12.11; macOS-15.8.1-arm64-arm-64bit.

Dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images), pinned at `c8c3cf72b3b724ce734259e4a086d30618a9198b`. Two-cat scene, car, cat portrait and receipt; source/derived hashes and transformation are retained in the [manifest](../../../benchmarks/vlm/hf-images.json). The dataset card declares CC BY-NC-SA 4.0. Downloaded image bytes are temporary inputs, not redistributed as project assets.

Batch 1, one warmup, five measured repetitions per workload, per-column median, greedy generation up to 128 emitted tokens. All counts and speeds use actual emitted IDs; EOS is respected. Twenty records contain 100 measured requests, excluding twenty warmups. The first exploratory S1 run is archived separately and is not pooled into the published medians.

## Standard image results

| Model | Precision | Input / emitted output tokens | Prefill TPS | Decode TPS | TTFT (ms) | Peak MLX (GiB) |
|---|---|---|---:|---:|---:|---:|
| [mlx-community/Qwen2-VL-2B-Instruct-4bit](../../benchmarks/vlm/results/V01-VLM-HF-S1.json) | Q4 | 359/112 | 279.68 | 67.28 | 1290.28 | 1.755 |
| [mlx-community/Qwen2.5-VL-3B-Instruct-4bit](../../benchmarks/vlm/results/V02-VLM-HF-S1.json) | Q4 | 359/79 | 215.86 | 41.19 | 1670.08 | 3.164 |
| [mlx-community/Qwen3.5-0.8B-4bit](../../benchmarks/vlm/results/V03-VLM-HF-S1.json) | Q4 | 284/106 | 717.75 | 68.78 | 401.16 | 1.129 |
| [mlx-community/LFM2-VL-450M-8bit](../../benchmarks/vlm/results/V04-VLM-HF-S1.json) | Q8 | 282/128 | 1739.05 | 154.33 | 169.97 | 1.163 |
| [LiquidAI/LFM2.5-VL-3B-MLX-4bit](../../benchmarks/vlm/results/V05-VLM-HF-S1.json) | Q4 | 281/101 | 300.73 | 52.01 | 942.38 | 2.801 |

Prefill is fused visual/language prefill, not pure text prefill. TTFT includes image decoding/preprocessing; chat-template rendering is outside the timed direct-model request. Memory includes resident model allocations after peak reset. Q4/Q8 denote published checkpoint formats; floating vision/projector modules retain their native weights. High-resolution, four-image and receipt workloads are separate JSON records, not mixed with S1 or the older synthetic-image protocol.

## Functional scope and limitations

### mlx-community/Qwen2-VL-2B-Instruct-4bit

Pinned revision: `01af461cdb9574acc09084a0ef94e216e142b085`. **CHECKPOINT_SMOKE_PASSED**. [Full Python/CLI/live HTTP/SSE evidence](V01-smoke.json).

- scope: bounded image-dependent functional spot check, not dataset accuracy
- primary_scene: Identifies two tabby cats resting on a pink/red couch with remote controls.
- changed_image: Identifies a turquoise Volkswagen Beetle in front of a yellow wall.
- multiturn: Answers that two cats are visible.
- receipt: Benchmark S4 reads document TD01167104, date 25/12/2018 and total 9.00.
- limitations: Scene descriptions contain minor pose/spatial inaccuracies; no general accuracy claim.

### mlx-community/Qwen2.5-VL-3B-Instruct-4bit

Pinned revision: `46d4cf06a06ffc1a766c214174f9cbed2f45bcab`. **CHECKPOINT_SMOKE_PASSED**. [Full Python/CLI/live HTTP/SSE evidence](V02-smoke.json).

- scope: bounded functional spot check, not dataset accuracy
- primary_scene: Identifies two striped cats and remote controls on a pink/red couch.
- changed_image: Identifies a light turquoise Volkswagen Beetle.
- multiturn: Correctly answers two cats.
- receipt: Reads TD01167104, date 25/12/2018 and total 9.00.
- limitations: Some cat-pose descriptions are inaccurate; no general accuracy claim.

### mlx-community/Qwen3.5-0.8B-4bit

Pinned revision: `da28692b5f139cb0ec58a356b437486b7dac7462`. **CHECKPOINT_SMOKE_PASSED**. [Full Python/CLI/live HTTP/SSE evidence](V03-smoke.json).

- scope: bounded image-dependent functional spot check; no dataset accuracy claim
- primary_scene: Identifies two tabby cats on a pink couch near remote controls.
- changed_image: Identifies a classic Volkswagen Beetle by a yellow building.
- multiturn: Correctly answers two cats.
- receipt: Date 25/12/2018 and total 9.00 are read; document number is misread as TD01167/104 and other fields contain character errors.
- limitations: Some spatial wording is inaccurate and OCR is not exact; S4 reaches the 128-token budget.

### mlx-community/LFM2-VL-450M-8bit

Pinned revision: `2422c0686f1f1c6267640bc1f069ff4e8b06b8d0`. **CHECKPOINT_SMOKE_PASSED**. [Full Python/CLI/live HTTP/SSE evidence](V04-smoke.json).

- scope: bounded image-dependent spot check; no dataset accuracy claim
- primary_scene: Identifies two resting cats on a pink/red couch or blanket.
- changed_image: Identifies a turquoise Volkswagen Beetle by a yellow building.
- multiturn: Correctly answers two cats.
- receipt: Date and total 9.00 are read, but document number is TD0116704 (missing a character) and the currency is incorrectly called RMB.
- limitations: Caption contains pose/detail errors; receipt transcription/currency is not reliable enough for financial use.

### LiquidAI/LFM2.5-VL-3B-MLX-4bit

Pinned revision: `7fad79d26092c1145aa8dde873896a07d3e96a3b`. **CHECKPOINT_SMOKE_PASSED**. [Full Python/CLI/live HTTP/SSE evidence](V05-smoke.json).

- scope: bounded image-dependent functional spot check, not dataset accuracy
- primary_scene: Identifies two sleeping striped cats on a pink couch near remote controls.
- changed_image: Identifies a mint-green Volkswagen Beetle by a yellow building.
- multiturn: Correctly answers two cats.
- receipt: Reads the date and main total 9.00, but document number contains an extra digit and rounded total is misread as 9.60.
- limitations: OCR is not exact; document identifiers and monetary fields need verification. No general accuracy claim.

All five smokes include authentication/alias checks, startup warmup, multi-turn image history, changed-image generation, early stream close/disconnect recovery, and repeated-request memory observations. Bounded image/OCR spot checks are not general accuracy or long-context qualification. Video, VLM tool calling, speculative VLM decode, managed image APC/paged KV and long-context validation remain outside scope.

## Validation and reproduction

Default regression: **2107 passed**, 70 skipped, 9 subtests passed. Targeted native suite: **49 passed**. Two existing TestClient deprecation warnings remain. Repository lint has zero new findings. Model source hashes remained unchanged across all twenty measured records, with one native source/runner cohort.

See the [methodology and exact commands](../../benchmarks/vlm/README.md#real-checkpoint-hf-documentation-image-protocol-2026-10-24). The runner verifies actual full checkpoint weight SHA-256 and processor/tokenizer provenance before executing. The README generator validates measurements from manifests even after downloaded images/models are removed.

## Cleanup

Task-owned model/image downloads and caches are removed after evidence validation. Final cleanup outcome is recorded in [report.json](report.json); unrelated user checkpoints and global caches are preserved.

Cleanup completed: **five downloaded models, image originals/derivatives and task HF caches removed**. Deleted 7,874,684,776 bytes of checkpoint weights and 7,960,583,030 total task bytes; the temporary root was verified absent. Only provenance, raw measurement JSON, smoke outputs and small execution summaries remain. Pre-existing user models and caches were preserved.
