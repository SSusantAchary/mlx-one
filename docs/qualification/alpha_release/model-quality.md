> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Model quality

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Thirty immutable applicable prompts per candidate/profile cover completion or ordinary chat, instruction, arithmetic, JSON, retrieval, repetition, explanation, EOS/budget behavior and multiturn. Three fixed prompts repeat natively. Actual prompt IDs, output IDs, budgets, finish reasons and strict rules are retained. Context overflow rejection and stop strings are exercised by the API suite; inputs are not silently truncated. The framework gate compares the independent reference. Shared task failures are model limitations; source-versus-primary changes are quantization behavior. Qwen3.5 is excluded from this new candidate fixture.

| Model/role | Profile | Native task score | Reference task score | Native-only failures | Repeats identical | Evidence |
|---|---|---|---|---|---|---|
| M01 source | stored | 30/30 | 30/30 | 0 | True | [raw](../archived-evidence.md) |
| M01 primary | stored | 30/30 | 30/30 | 0 | True | [raw](../archived-evidence.md) |
| M02 source | stored | 22/30 | 22/30 | 0 | True | [raw](../archived-evidence.md) |
| M02 primary | affine_q8 | 22/30 | 22/30 | 0 | True | [raw](../archived-evidence.md) |
| M03 source | stored | 16/30 | 16/30 | 0 | True | [raw](../archived-evidence.md) |
| M03 primary | affine_q8 | 16/30 | 16/30 | 0 | True | [raw](../archived-evidence.md) |
| M04 source | stored | 13/30 | 13/30 | 0 | True | [raw](../archived-evidence.md) |
| M04 primary | affine_q4 | 14/30 | 14/30 | 0 | True | [raw](../archived-evidence.md) |
| M06 source | stored | 19/30 | 19/30 | 0 | True | [raw](../archived-evidence.md) |
| M06 primary | affine_q4 | 20/30 | 20/30 | 0 | True | [raw](../archived-evidence.md) |
| M07 source | stored | 17/30 | 17/30 | 0 | True | [raw](../archived-evidence.md) |
| M07 primary | affine_q4 | 16/30 | 16/30 | 0 | True | [raw](../archived-evidence.md) |
| M08 source | stored | 19/30 | 19/30 | 0 | True | [raw](../archived-evidence.md) |
| M08 primary | affine_q4 | 17/30 | 17/30 | 0 | True | [raw](../archived-evidence.md) |
| M09 source | stored | 17/30 | 17/30 | 0 | True | [raw](../archived-evidence.md) |
| M09 primary | stored | 17/30 | 17/30 | 0 | True | [raw](../archived-evidence.md) |
| M10 source | stored | 19/30 | 19/30 | 0 | True | [raw](../archived-evidence.md) |
| M10 primary | affine_q4 | 15/30 | 15/30 | 0 | True | [raw](../archived-evidence.md) |

[quality-fixture-v1.json](../archived-evidence.md); [quality-fixture-freeze.json](../archived-evidence.md).
