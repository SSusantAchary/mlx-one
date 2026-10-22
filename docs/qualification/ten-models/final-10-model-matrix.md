> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Final ten-checkpoint matrix

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

**Alpha: NO-GO.** Native families implemented 8/8; strict whole-port equivalence validated 7/8 families / 9/10 checkpoints. Source-qualified 0/10; primary-qualified 0/10; fully qualified 0/10. API 58/58 applicable source combinations pass. Port equivalence requires cross logits/state plus independent controls and remains distinct from reference/native self-consistency and full release qualification.

| ID | Family | Port cross math/state | Source qualified | Cache qualified | Primary profile / qualified | Quality primary N/R | API | Fully qualified |
|---|---|---|---|---|---|---|---|---|
| M01 | gpt2 | PASS | NO | NO | stored / NO | 2/2 / 2/2 | PASS | NO |
| M02 | qwen2 | PASS | NO | NO | affine_q8 / NO | 6/8 / 6/8 | PASS | NO |
| M03 | qwen3 | PASS | NO | NO | affine_q8 / NO | 5/8 / 5/8 | PASS | NO |
| M04 | lfm2 | PASS | NO | NO | affine_q4 / NO | 4/8 / 4/8 | PASS | NO |
| M05 | qwen3_5 | FAIL_NUMERICAL | NO | NO | affine_q4 / NO | 5/8 / 5/8 | PASS | NO |
| M06 | llama | PASS | NO | NO | affine_q4 / NO | 5/8 / 5/8 | PASS | NO |
| M07 | llama | PASS | NO | NO | affine_q4 / NO | 7/8 / 7/8 | PASS | NO |
| M08 | qwen2 | PASS | NO | NO | affine_q4 / NO | 5/8 / 5/8 | PASS | NO |
| M09 | gemma3_text | PASS | NO | NO | stored / NO | 7/8 / 7/8 | PASS | NO |
| M10 | phi3 | PASS | NO | NO | affine_q4 / NO | 5/8 / 5/8 | PASS | NO |

1822 passed, 54 skipped, 3 warnings in 63.12s (0:01:03). Native acceptance: 353 passed, 2 warnings in 7.70s. Entire native suite: 8 failed, 2066 passed, 18 skipped, 3 warnings in 99.55s (0:01:39). Its eight failures reproduce in the hash-verified entry source (nine entry failures); no newly failing test identities. Ruff 0 new / 65 inherited. Fresh wheel PASS.

[Implementation](reference-first-implementation.md), [provenance](reference-source-provenance.md), [source math](source-weight-parity.md), [self-consistency](reference-self-consistency.md), [cache state](cache-state-parity.md), [quantization](quantization-qualification.md), [quality](generation-quality.md), [APIs](api-regression.md), [hardware](hardware-reference-first.md), [blockers](../archived-evidence.md), [machine-readable summary](../archived-evidence.md).
