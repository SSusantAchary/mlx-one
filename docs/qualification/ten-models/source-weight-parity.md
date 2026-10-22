> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Source-weight parity

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

The source phase ran before primary quantization. A final ten-worker revalidation records the exact finalized production file hashes. Each worker uses the original local reference, 16 prompt tokens plus 16 forced continuation tokens, all 17 logit boundaries, committed states, whole/chunk/reset and independent greedy generation. The separate fresh-prefix oracle covers every prefix through 64 and the required partitions. Production changes between the initial source oracle and final revalidation were adapter/metadata/cache-version corrections; final raw gates remain failed.

| ID | Checkpoint | Entry max cross error | Port max cross error | Cross-state failures | Source qualified | Evidence |
|---|---|---:|---:|---:|---|---|
| M01 | openai-community/gpt2 | 0.007543563843 | 0 | 0 | NO | [M01.json](../archived-evidence.md) |
| M02 | Qwen/Qwen2.5-0.5B-Instruct | 0.4921875 | 0 | 0 | NO | [M02.json](../archived-evidence.md) |
| M03 | Qwen/Qwen3-0.6B | 0.40625 | 0 | 0 | NO | [M03.json](../archived-evidence.md) |
| M04 | LiquidAI/LFM2-350M | 0.28125 | 0 | 0 | NO | [M04.json](../archived-evidence.md) |
| M05 | Qwen/Qwen3.5-0.8B | 0.3125 | 0.3125 | 42 | NO | [M05.json](../archived-evidence.md) |
| M06 | HuggingFaceTB/SmolLM2-1.7B-Instruct | 0.421875 | 0 | 0 | NO | [M06.json](../archived-evidence.md) |
| M07 | meta-llama/Llama-3.2-1B-Instruct | 0.1875 | 0 | 0 | NO | [M07.json](../archived-evidence.md) |
| M08 | Qwen/Qwen2.5-1.5B-Instruct | 0.34375 | 0 | 0 | NO | [M08.json](../archived-evidence.md) |
| M09 | google/gemma-3-1b-it | 0 | 0 | 0 | NO | [M09.json](../archived-evidence.md) |
| M10 | microsoft/Phi-3-mini-4k-instruct | 0 | 0 | 0 | NO | [M10.json](../archived-evidence.md) |

Nine ports match the reference exactly at all tested source full/cached logits and committed states. The entry implementation achieves this across the 17-boundary/state comparison for two checkpoints. Qwen3.5 retains HF raw-offset FP32 norms, while MLX-LM folds weights in BF16. Its first cross difference is layer 0 input_layernorm; first traced failure at prefix 1 is layer 3 input_layernorm. This is not declared benign or the sole explanation of every downstream failure. The original frozen full-model gate remains FAIL.
