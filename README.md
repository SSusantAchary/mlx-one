<p align="center"><img src="mlx-one.png" alt="mlx-one" width="900"></p>

# mlx-one

**Train locally. Prove it on Apple Silicon. Scale when needed.**

MLX-One provides native MLX model execution and lifecycle tools for Apple Silicon. The Python Engine, CLI and HTTP server share the project’s model implementations. Tokenizer assets use Hugging Face formats; production text inference does not require MLX-LM or Transformers.

**0.1.0a1 is an unpublished development candidate.** Default model admission follows the [functional architecture and checkpoint policy](docs/model-admission.md). Publication approval remains pending. Architecture implementation and exact-checkpoint smoke are separate statuses; optional numerical research does not veto functional admission.

The qualification host is an Apple M4 with 32 GB, macOS 15.7.9, Python 3.12.11 and MLX/Metal 0.32.2. The packaged MLX runtime is pinned to that tested pair. Python >=3.10 is the metadata requirement; other Python and device combinations have not received this release certification. The runtime wheels require Apple Silicon macOS 15 or later.

Install the development checkout in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/alpha-tested-constraints.txt -e ".[server]"
mlx-one --version
```

For an experimental local checkpoint already downloaded on the machine:

```python
from mlx_one import CacheConfig, Engine

with Engine("/path/to/checkpoint", offline=True,
            inference_mode="reference_dense",
            cache_config=CacheConfig(prefix_cache_enabled=False)) as engine:
    print(engine.generate("The city was", max_tokens=32).text)
```

```sh
mlx-one generate /path/to/checkpoint "The city was" --offline --max-tokens 32
mlx-one serve /path/to/checkpoint --offline --inference-mode reference_dense
```

```sh
curl http://127.0.0.1:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID_FROM_V1_MODELS","prompt":"The city was","max_tokens":32,"temperature":0,"stream":true}'
```

Chat-enabled checkpoints also accept `POST /v1/chat/completions` with `messages`. See [runnable Python and HTTP examples](examples/alpha_release/) and the [HTTP contract](docs/api/openai-compatibility.md). `/v1/models` describes the loaded model; `/v1/runtime` reports its actual execution/state configuration and release status. Optional authentication uses the server’s existing API-key configuration.

## Supported Models

The bounded runtime scope is B1 `reference_dense`, one startup model, FP16 KV and existing sliding/convolution state. The ten entries below retain their exact revisions and historical functional evidence. `EXPERIMENTAL_SUPPORTED` describes that executed scope; publication approval and independent user validation remain separate. A dash in context means no proven practical range.

| Model / exact source revision | Task | Primary weights | State | Qualified context | Python | CLI | HTTP/chat | SSE | Backend | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| [openai-community/gpt2](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e) `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Completion | float32 | FP16 KV | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) `7ae557604adf67be50417f59c2c2f167def9a775` | Completion/chat | affine_q8 | FP16 KV | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) `c1899de289a04d12100db370d81485cdf75e47ca` | Completion/chat | affine_q8 | FP16 KV | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M/tree/0b6ea934a94b880d1048cbcba45a0d81ca721eb0) `0b6ea934a94b880d1048cbcba45a0d81ca721eb0` | Completion/chat | affine_q4 | FP16 KV + convolution | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) `2fc06364715b967f1860aea9cf38778875588b17` | Completion/chat | affine_q4 | FP16 KV + FP32 recurrent | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | Experimental; excluded |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) `31b70e2e869a7173562077fd711b654946d38674` | Completion/chat | affine_q4 | FP16 KV | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/9213176726f574b556790deb65791e0c5aa438b6) `9213176726f574b556790deb65791e0c5aa438b6` | Completion/chat | affine_q4 | FP16 KV | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Completion/chat | affine_q4 | FP16 KV | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it/tree/dcc83ea841ab6100d6b47a070329e1ba4cf78752) `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | Completion/chat | bfloat16 | FP16 KV + sliding | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |
| [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/f39ac1d28e925b323eae81227eaba4464caced4e) `f39ac1d28e925b323eae81227eaba4464caced4e` | Completion/chat | affine_q4 | FP16 KV + sliding | — | Experimental | Experimental | Experimental | Experimental | Native MLX dense B1 | EXPERIMENTAL_SUPPORTED; bounded functional smoke |

<!-- model-performance:start -->
## Model Performance

Actual **direct-model inference** measurements; experimental, batch size 1. Python Engine, CLI and HTTP serving throughput are separate measurement types and are not represented by these numbers. Parameters are nominal model sizes.

**Historical 128-input / 16-output protocol.** These archived runs are not the new standardized 128-input / 128-output results. Source and primary precision measurements are listed separately; “source” means the stored weights.

**historical-128-16** — Apple M4, 32 GB; macOS-15.7.9-arm64-arm-64bit; Python 3.12.11; MLX 0.32.2 / Metal 0.32.2. Backend: native MLX diagnostic dense; full sequence logits; production_fp16 KV. Actual input: 128 tokens; actual output: 16 tokens per repetition. 1 shape warmup; 3 measured repetitions; per-column median. FP16 KV; fresh state, no prefix reuse. TTFT includes prefill and greedy selection of the first token. Peak MLX resets before each request after load/conversion/warmup; it includes resident model allocations. Implementation `5abf36ab054c`; runner `e33774669c37`.

| Model name | Parameters | Quantization/precision | Prefill tokens/sec | Decode tokens/sec | Time to first token (ms) | Peak MLX memory (GiB) |
|---|---|---|---:|---:|---:|---:|
| [openai-community/gpt2](docs/benchmarks/historical/M01-stored-performance.json) | 124M | FP32 | 6722.29 | 118.04 | 19.04 | 0.594 |
| [Qwen/Qwen2.5-0.5B-Instruct](docs/benchmarks/historical/M02-affine_q8-performance.json) | 0.5B | Affine Q8 (group 64) | 2001.96 | 83.48 | 63.94 | 0.632 |
| [Qwen/Qwen2.5-0.5B-Instruct](docs/benchmarks/historical/M02-stored-performance.json) | 0.5B | BF16 (source) | 1801.11 | 50.49 | 71.07 | 1.068 |
| [Qwen/Qwen3-0.6B](docs/benchmarks/historical/M03-affine_q8-performance.json) | 0.6B | Affine Q8 (group 64) | 1275.02 | 65.47 | 100.39 | 0.816 |
| [Qwen/Qwen3-0.6B](docs/benchmarks/historical/M03-stored-performance.json) | 0.6B | BF16 (source) | 892.22 | 33.95 | 143.46 | 1.312 |
| [LiquidAI/LFM2-350M](docs/benchmarks/historical/M04-affine_q4-performance.json) | 350M | Affine Q4 (group 64) | 3180.66 | 185.81 | 40.24 | 0.313 |
| [LiquidAI/LFM2-350M](docs/benchmarks/historical/M04-stored-performance.json) | 350M | BF16 (source) | 3169.32 | 91.87 | 40.39 | 0.783 |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](docs/benchmarks/historical/M06-affine_q4-performance.json) | 1.7B | Affine Q4 (group 64) | 384.06 | 55.69 | 333.28 | 1.202 |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](docs/benchmarks/historical/M06-stored-performance.json) | 1.7B | BF16 (source) | 309.63 | 19.21 | 413.39 | 3.390 |
| [meta-llama/Llama-3.2-1B-Instruct](docs/benchmarks/historical/M07-affine_q4-performance.json) | 1B | Affine Q4 (group 64) | 374.73 | 65.36 | 341.58 | 0.886 |
| [meta-llama/Llama-3.2-1B-Instruct](docs/benchmarks/historical/M07-stored-performance.json) | 1B | BF16 (source) | 572.81 | 29.75 | 223.46 | 2.455 |
| [Qwen/Qwen2.5-1.5B-Instruct](docs/benchmarks/historical/M08-affine_q4-performance.json) | 1.5B | Affine Q4 (group 64) | 411.23 | 53.52 | 311.26 | 1.034 |
| [Qwen/Qwen2.5-1.5B-Instruct](docs/benchmarks/historical/M08-stored-performance.json) | 1.5B | BF16 (source) | 375.86 | 21.46 | 340.55 | 3.044 |
| [google/gemma-3-1b-it](docs/benchmarks/historical/M09-stored-performance.json) | 1B | BF16 | 675.56 | 32.17 | 189.47 | 2.028 |
| [microsoft/Phi-3-mini-4k-instruct](docs/benchmarks/historical/M10-affine_q4-performance.json) | 3.8B | Affine Q4 (group 64) | 159.64 | 29.92 | 801.79 | 2.357 |
| [microsoft/Phi-3-mini-4k-instruct](docs/benchmarks/historical/M10-stored-performance.json) | 3.8B | BF16 (source) | 126.13 | 10.87 | 1014.81 | 7.256 |

**standard-128-128** — Apple M4, 32 GB; macOS-15.8.1-arm64-arm-64bit; Python 3.12.11; MLX 0.32.2 / Metal 0.32.2. Backend: native MLX direct dense; last_only logits; production_fp16 KV. Actual input: 128 tokens; actual output: 128 tokens per repetition. 1 shape warmup; 3 measured repetitions; per-column median. FP16 KV; fresh state, no prefix reuse. TTFT includes prefill and greedy selection of the first token. Peak MLX resets before each request after load/conversion/warmup; it includes resident model allocations. Implementation `aad603ce84e2`; runner `b4008453bc42`.

| Model name | Parameters | Quantization/precision | Prefill tokens/sec | Decode tokens/sec | Time to first token (ms) | Peak MLX memory (GiB) |
|---|---|---|---:|---:|---:|---:|
| [openai-community/gpt2](docs/benchmarks/runs/M01-direct-128-128.json) | 124M | FP32 | 4057.67 | 96.48 | 31.55 | 0.570 |

**Standardized 128/128 coverage: 1 / 13 primary configurations.** Remaining runs are pending; see the coverage report.

**Missing primary-precision performance:** `Qwen/Qwen3.5-0.8B`, `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`, `mlx-community/gpt-oss-20b-MXFP4-Q4`, `manjunathshiva/gpt-oss-20b-tq3`. No speed or memory figures are inferred for these checkpoints.

See the [methodology and manual commands](docs/benchmarks/README.md), [checkpoint/precision inventory](docs/benchmarks/inventory.md), and [coverage report](docs/benchmarks/coverage.md). Full samples, output token IDs, immutable revisions, source hashes and hardware evidence live in [docs/benchmarks](docs/benchmarks/).
<!-- model-performance:end -->

## Experimental model notes

Q4/Q8 candidates are MLX affine conversions of the pinned source weights with group size 64. Source precision and conversion results are evaluated separately. Gemma3 and Phi3 retain bounded sliding state; LFM2 retains its causal convolution history. Qwen3.5 remains `EXPERIMENTAL_NOT_QUALIFIED`: a fresh stored-source 8-token generation and native continuation/reset pass at context 512, while its converted Q4 and previously broken paths remain unqualified. See the [Qwen3.5 limitations](src/mlx_one/models/vision_language/qwen3_5/README.md) and [Gemma3 guide](src/mlx_one/models/language/gemma3/README.md). Gemma3 is admitted under the same functional policy; historical numerical differences and repeated/weak answers are disclosed limitations, not numerical admission vetoes.

The [historical release proof](docs/qualification/archived-evidence.md) is preserved locally with verified archive hashes; compact public summaries retain its original outcomes. The [current admission policy](docs/model-admission.md), [native MoE report](docs/native-moe/implementation-report.md), and packaged support manifest describe current functional status.

Two additional native families cover three planned checkpoints and are **CODE_IMPLEMENTED / SYNTHETIC_TESTED**. None of these three checkpoints was downloaded, loaded, generated from, or manually validated in this task.

| Exact checkpoint | Architecture | Weight formats | Status | Checkpoint validation |
|---|---|---|---|---|
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` @ `667408d494bde191c2cb9dd4f4549827816b9275` | `lfm2_moe` | affine Q4, tied embedding Q6, FP32 router | ARCHITECTURE_IMPLEMENTED | NOT_DOWNLOADED_NOT_SMOKE_TESTED; MANUAL_TEST_PENDING |
| `mlx-community/gpt-oss-20b-MXFP4-Q4` @ `f356f2747216d7e98fee755df25987459fc19089` | `gpt_oss` | native MXFP4 experts, module affine Q4/Q8 | ARCHITECTURE_IMPLEMENTED | NOT_DOWNLOADED_NOT_SMOKE_TESTED; MANUAL_TEST_PENDING |
| `manjunathshiva/gpt-oss-20b-tq3` @ `20ba1835e625264e07366d215933fe31a9940a6a` | `gpt_oss` | TQ3 / TurboQuant 3-bit, floating embedding/router | QUANTIZATION_BACKEND_TESTED | NOT_DOWNLOADED_NOT_SMOKE_TESTED; PENDING_USER_DOWNLOAD |

All three configure a maximum of **131072 positions**. This is not a demonstrated practical context on M4/32GB. Start at 128–1024 tokens. Use the [local-path download, CLI, Python, authenticated HTTP/SSE and metrics runbook](docs/native-moe/manual-inference.md). GPT-OSS renders only assistant final-channel text and honors all checkpoint EOS IDs; a short budget may end before any final text. The runtime uses native MLX 0.32.2, without production MLX-LM/Transformers imports. Liquid weights use `lfm1.0`; GPT-OSS weights declare Apache-2.0. The package redistributes no weights.

Native TQ3 uses packed Metal GEMV/GEMM and selected-expert projections without expanding a model or expert bank to FP16. See the [TQ3 implementation and measured operator report](docs/implementation/turboquant-tq3/performance-report.md) and [fresh-wheel/manual benchmark guide](docs/implementation/turboquant-tq3/manual-validation.md). Synthetic operator results do not establish full-model throughput; the 90% warm-decode target is pending the owner’s checkpoint run.

Continuous batching, paging, INT8 paged KV, speculation, SSD persistence, untested context lengths and unqualified precision conversions remain experimental. Other registered text, vision, audio and retrieval architectures are retained; this candidate scope does not certify them. Small-model instruction, arithmetic and strict JSON limitations are recorded separately from framework defects.

License: [Apache-2.0](LICENSE), with retained MIT notices for MLX-LM-derived mathematics. See [upstream notices](src/mlx_one/models/UPSTREAM_LICENSE), [architecture documentation](docs/architecture/) and [changelog](CHANGELOG.md).

Repository documentation: [native architecture](docs/architecture/mlx-one-v0.1.0-revamp.md), [inference engine and roadmap](docs/architecture/inference-engine.md), and [cleanup/archive policy](docs/qualification/archived-evidence.md).
