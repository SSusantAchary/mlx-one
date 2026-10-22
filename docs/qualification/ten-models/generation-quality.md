> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Generation quality

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

The original prompts and expected answers are unchanged. Fixture v2 budgets completion 32, arithmetic 256, instruction 32, JSON 128, explanation 128, repetition 32, retrieval 64 and multiturn 64 tokens. Exact JSON cannot be wrapped in code fences. GPT-2 has two applicable base-completion cases and six UNSUPPORTED_TASK cases. Scores are bounded fixture evidence, not general model intelligence.

Native/reference quality runs use matched KV storage: BF16 for stored BF16 models and FP16 for quantized models. Public GPT-2 FP32 uses matched FP16 KV via an explicit reference storage view; its raw FP32-cache mathematical gates remain separate. Reference code itself is unchanged. Both paths use identical prompt token IDs, EOS IDs, templates, budgets and greedy policy.

| ID | Source native / reference | Primary native / reference | Primary fixture failures |
|---|---|---|---|
| M01 | 2/2 / 2/2 | 2/2 / 2/2 | none |
| M02 | 6/8 / 6/8 | 6/8 / 6/8 | arithmetic: FAIL_ARITHMETIC, json: FAIL_JSON |
| M03 | 5/8 / 5/8 | 5/8 / 5/8 | arithmetic: FAIL_INSTRUCTION, instruction: FAIL_INSTRUCTION, json: FAIL_JSON |
| M04 | 4/8 / 4/8 | 4/8 / 4/8 | arithmetic: FAIL_INSTRUCTION, instruction: TRUNCATED, json: FAIL_JSON, bounded-retrieval: FAIL_RETRIEVAL |
| M05 | 7/8 / 7/8 | 5/8 / 5/8 | arithmetic: FAIL_ARITHMETIC, instruction: FAIL_INSTRUCTION, json: FAIL_JSON |
| M06 | 5/8 / 5/8 | 5/8 / 5/8 | arithmetic: FAIL_ARITHMETIC, bounded-retrieval: FAIL_RETRIEVAL, multiturn: FAIL_INSTRUCTION |
| M07 | 7/8 / 7/8 | 7/8 / 7/8 | json: FAIL_JSON |
| M08 | 7/8 / 7/8 | 5/8 / 5/8 | json: FAIL_JSON, bounded-retrieval: FAIL_RETRIEVAL, multiturn: FAIL_INSTRUCTION |
| M09 | 7/8 / 7/8 | 7/8 / 7/8 | json: FAIL_JSON |
| M10 | 6/8 / 6/8 | 5/8 / 5/8 | instruction: TRUNCATED, json: FAIL_JSON, multiturn: FAIL_INSTRUCTION |

No native-only scored task failures occur in this fixture. This does not erase failed numerical/state gates. Qwen3.5 text/token differences, including trailing whitespace, are retained. Truncation is explicit; Phi3 Q4's instruction failure reaches its 32-token budget. Source correctness, quantization degradation, strict instruction/JSON formatting and framework behavior remain separate categories. [All raw generations](../archived-evidence.md).
