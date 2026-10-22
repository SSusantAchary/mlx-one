# Current model support matrix

Derived from the packaged support manifest. Alpha publication remains **PENDING**.
Functional experimental support does not imply alpha release approval, broad context qualification
or exhaustive numerical equivalence.

| Checkpoint | Architecture | Precision | Status | Full-checkpoint evidence |
|---|---|---|---|---|
| `openai-community/gpt2` | `gpt2` | float32 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `Qwen/Qwen2.5-0.5B-Instruct` | `qwen2` | affine_q8 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `Qwen/Qwen3-0.6B` | `qwen3` | affine_q8 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `LiquidAI/LFM2-350M` | `lfm2` | affine_q4 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `Qwen/Qwen3.5-0.8B` | `qwen3_5` | affine_q4 | EXPERIMENTAL_NOT_QUALIFIED | MODEL_SMOKE_TESTED |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `llama` | affine_q4 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `meta-llama/Llama-3.2-1B-Instruct` | `llama` | affine_q4 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `Qwen/Qwen2.5-1.5B-Instruct` | `qwen2` | affine_q4 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `google/gemma-3-1b-it` | `gemma3_text` | bfloat16 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `microsoft/Phi-3-mini-4k-instruct` | `phi3` | affine_q4 | EXPERIMENTAL_SUPPORTED | MODEL_SMOKE_TESTED |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | `lfm2_moe` | mixed_affine_q4_embedding_q6_router_fp32 | ARCHITECTURE_IMPLEMENTED | NOT_DOWNLOADED_NOT_SMOKE_TESTED |
| `mlx-community/gpt-oss-20b-MXFP4-Q4` | `gpt_oss` | mixed_mxfp4_affine_q4_q8 | ARCHITECTURE_IMPLEMENTED | NOT_DOWNLOADED_NOT_SMOKE_TESTED |
| `manjunathshiva/gpt-oss-20b-tq3` | `gpt_oss` | TQ3 / TurboQuant 3-bit | QUANTIZATION_BACKEND_TESTED | NOT_DOWNLOADED_NOT_SMOKE_TESTED |

See the [admission policy](../model-admission.md) and [archived historical evidence](../qualification/archived-evidence.md).
The two large MoE checkpoints are architecture-only candidates; TQ3 has a synthetically tested
quantization backend. None of these three large checkpoints was downloaded or smoke-tested.
Qwen3.5 remains experimental and unqualified. Existing known quality/numerical limitations remain
disclosed in the README and historical summaries. No support status was promoted by cleanup.
