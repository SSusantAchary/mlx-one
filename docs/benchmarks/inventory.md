# Text inference checkpoint inventory

Primary means the support manifest’s selected configuration, not certification. Q4/Q8 conversions use group size 64. Source BF16/FP32 runs are distinct. Parameter sizes are nominal labels; quantized bytes are not parameter counts.


| ID / checkpoint | Revision | Parameters | Primary precision | Available benchmark profiles | Functional status |
|---|---|---|---|---|---|
| M01 `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 124M | FP32 | stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M02 `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | 0.5B | Affine Q8 (group 64) | affine_q8, stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M03 `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 0.6B | Affine Q8 (group 64) | affine_q8, stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M04 `LiquidAI/LFM2-350M` | `0b6ea934a94b880d1048cbcba45a0d81ca721eb0` | 350M | Affine Q4 (group 64) | affine_q4, stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M05 `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 0.8B | Affine Q4 (group 64) | None | EXPERIMENTAL_NOT_QUALIFIED / MODEL_SMOKE_TESTED |
| M06 `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | 1.7B | Affine Q4 (group 64) | affine_q4, stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M07 `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | 1B | Affine Q4 (group 64) | affine_q4, stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M08 `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 1.5B | Affine Q4 (group 64) | affine_q4, stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M09 `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | 1B | BF16 | stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M10 `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` | 3.8B | Affine Q4 (group 64) | affine_q4, stored | EXPERIMENTAL_SUPPORTED / MODEL_SMOKE_TESTED |
| M11 `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | `667408d494bde191c2cb9dd4f4549827816b9275` | 24B / 2B active | Mixed affine Q4 / embedding Q6 / FP32 router | None | ARCHITECTURE_IMPLEMENTED / NOT_DOWNLOADED_NOT_SMOKE_TESTED |
| M12 `mlx-community/gpt-oss-20b-MXFP4-Q4` | `f356f2747216d7e98fee755df25987459fc19089` | 20B / 3.6B active | Mixed MXFP4 / affine Q4–Q8 | None | ARCHITECTURE_IMPLEMENTED / NOT_DOWNLOADED_NOT_SMOKE_TESTED |
| M13 `manjunathshiva/gpt-oss-20b-tq3` | `20ba1835e625264e07366d215933fe31a9940a6a` | 20B / 3.6B active | TurboQuant TQ3 | None | QUANTIZATION_BACKEND_TESTED / NOT_DOWNLOADED_NOT_SMOKE_TESTED |
