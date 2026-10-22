> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Generation quality requalification

Cohort `numerical-correction-20261020-1792496951619201000`. Fixture v2 preserves all eight original prompts and exact expected answers. It changes output budgets and failure classification only. Native public Engine and an independent local-reference greedy decoder are executed separately for each precision. Actual prompts, SHA256 identities, token IDs, template options, budgets and finish reasons are in the raw quality records.

| Task | Budget | Prompt / contract |
|---|---:|---|
| completion | 32 | The capital of France is |
| arithmetic | 256 | Compute 17 + 25. Return only the number. |
| instruction | 32 | Return exactly YES and nothing else. |
| json | 128 | Return only a JSON object with exactly two keys: name with value Lin, and count with value 2. |
| explanation | 128 | Explain what an API does in one sentence. |
| repetition | 32 | Continue the pattern briefly: blue blue blue |
| bounded-retrieval | 64 | The secret value is orchid-17. {filler} What is the secret value? Return only the value. |
| multiturn | 64 | [{"role": "user", "content": "My name is Lin."}, {"role": "assistant", "content": "Hello, Lin."}, {"role": "user", "content": "What is my name? Return only the name."}] |

Retrieval expands `{filler}` into 128 repetitions of `background `. JSON is parsed as generated, without removing code fences. Exact answers permit surrounding whitespace only. GPT-2 is a base completion model: six instruction tasks are UNSUPPORTED_TASK, not passing or skipped successes. Nonempty completion/explanation/repetition checks are limited surface checks; unique-word ratios are diagnostic rather than a broad semantic or repetition-quality claim.

Native public generation uses default FP16 KV; source reference generation uses activation-precision KV. This difference is explicitly retained in each quality record, so quality comparison does not replace same-state numerical parity. No failure is blamed automatically on quantization or truncation.

| ID | Source native/ref | Primary precision | Primary native/ref | Quantization-only failed cases |
|---|---|---|---|---|
| M01 | 2/2 / 2/2 | float32 | 2/2 / 2/2 | none observed |
| M02 | 6/8 / 6/8 | affine_q8 | 6/8 / 6/8 | none observed |
| M03 | 5/8 / 5/8 | affine_q8 | 5/8 / 5/8 | none observed |
| M04 | 4/8 / 4/8 | affine_q4 | 4/8 / 4/8 | none observed |
| M05 | 7/8 / 7/8 | affine_q4 | 5/8 / 5/8 | arithmetic, instruction |
| M06 | 5/8 / 5/8 | affine_q4 | 5/8 / 5/8 | none observed |
| M07 | 7/8 / 7/8 | affine_q4 | 7/8 / 7/8 | none observed |
| M08 | 7/8 / 7/8 | affine_q4 | 5/8 / 5/8 | bounded-retrieval, multiturn |
| M09 | 7/8 / 7/8 | bfloat16 | 7/8 / 7/8 | none observed |
| M10 | 6/8 / 6/8 | affine_q4 | 5/8 / 5/8 | instruction |

## M01 stored (stored)

Template present: `False`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | N/A | N/A | 0 | N/A | UNSUPPORTED_TASK | UNSUPPORTED_TASK |
| instruction | N/A | N/A | 0 | N/A | UNSUPPORTED_TASK | UNSUPPORTED_TASK |
| json | N/A | N/A | 0 | N/A | UNSUPPORTED_TASK | UNSUPPORTED_TASK |
| explanation | N/A | N/A | 0 | N/A | UNSUPPORTED_TASK | UNSUPPORTED_TASK |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | N/A | N/A | 0 | N/A | UNSUPPORTED_TASK | UNSUPPORTED_TASK |
| multiturn | N/A | N/A | 0 | N/A | UNSUPPORTED_TASK | UNSUPPORTED_TASK |

## M02 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | 43b7d959a370bc0e | 256 | 10 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| instruction | f65fd7843ce3d62a | 32 | 1 | stop | PASS | PASS |
| json | f08df661bdcc804b | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 68cfecef6ab358d5 | 128 | 28 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | f5a2897003a7a5d1 | 64 | 5 | stop | PASS | PASS |
| multiturn | ec51a550ea046748 | 64 | 1 | stop | PASS | PASS |

Failure `arithmetic`: **FAIL_INSTRUCTION**, finish `stop`; native text "17 + 25 = 42". Reference text "17 + 25 = 42". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M02 primary (affine_q8)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | 43b7d959a370bc0e | 256 | 2 | stop | FAIL_ARITHMETIC | FAIL_ARITHMETIC |
| instruction | f65fd7843ce3d62a | 32 | 1 | stop | PASS | PASS |
| json | f08df661bdcc804b | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 68cfecef6ab358d5 | 128 | 28 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | f5a2897003a7a5d1 | 64 | 5 | stop | PASS | PASS |
| multiturn | ec51a550ea046748 | 64 | 1 | stop | PASS | PASS |

Failure `arithmetic`: **FAIL_ARITHMETIC**, finish `stop`; native text "32". Reference text "32". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M03 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | ed59fe219a834ae7 | 256 | 10 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| instruction | 8fd39ab149922d90 | 32 | 2 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| json | 2d6db1603b62aee3 | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | ab53b16a1964bb72 | 128 | 26 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 23a0cf4c6f981c8e | 64 | 5 | stop | PASS | PASS |
| multiturn | b54d20bd41372475 | 64 | 1 | stop | PASS | PASS |

Failure `arithmetic`: **FAIL_INSTRUCTION**, finish `stop`; native text "17 + 25 = 42". Reference text "17 + 25 = 42". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `instruction`: **FAIL_INSTRUCTION**, finish `stop`; native text "YES.". Reference text "YES.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M03 primary (affine_q8)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | ed59fe219a834ae7 | 256 | 10 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| instruction | 8fd39ab149922d90 | 32 | 2 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| json | 2d6db1603b62aee3 | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | ab53b16a1964bb72 | 128 | 26 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 23a0cf4c6f981c8e | 64 | 5 | stop | PASS | PASS |
| multiturn | b54d20bd41372475 | 64 | 1 | stop | PASS | PASS |

Failure `arithmetic`: **FAIL_INSTRUCTION**, finish `stop`; native text "17 + 25 = 42". Reference text "17 + 25 = 42". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `instruction`: **FAIL_INSTRUCTION**, finish `stop`; native text "YES.". Reference text "YES.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M04 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | 19eb38e17e57acb2 | 256 | 7 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| instruction | 9880ff2ef1f7c864 | 32 | 9 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| json | 4917f0af6b1f5435 | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 03904b590467fe23 | 128 | 25 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 75e564236245d4d5 | 64 | 59 | stop | FAIL_RETRIEVAL | FAIL_RETRIEVAL |
| multiturn | d9e307c02887ff98 | 64 | 1 | stop | PASS | PASS |

Failure `arithmetic`: **FAIL_INSTRUCTION**, finish `stop`; native text "17 + 25 = 42". Reference text "17 + 25 = 42". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `instruction`: **FAIL_INSTRUCTION**, finish `stop`; native text "Yes, I will return exactly YES.". Reference text "Yes, I will return exactly YES.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `bounded-retrieval`: **FAIL_RETRIEVAL**, finish `stop`; native text "The secret value is **17**. \n\nIt's a common practice in cryptography to keep a secret key or a specific number like orchid-17 safe and sound.  The exact nature of the value itself isn't disclosed without further context, but 17 is the known secret.". Reference text "The secret value is **17**. \n\nIt's a common practice in cryptography to keep a secret key or a code value, and in this case, it's orchid-17.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M04 primary (affine_q4)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | 19eb38e17e57acb2 | 256 | 7 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| instruction | 9880ff2ef1f7c864 | 32 | 32 | length | TRUNCATED | TRUNCATED |
| json | 4917f0af6b1f5435 | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 03904b590467fe23 | 128 | 29 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 75e564236245d4d5 | 64 | 12 | stop | FAIL_RETRIEVAL | FAIL_RETRIEVAL |
| multiturn | d9e307c02887ff98 | 64 | 1 | stop | PASS | PASS |

Failure `arithmetic`: **FAIL_INSTRUCTION**, finish `stop`; native text "17 + 25 = 42". Reference text "17 + 25 = 42". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `instruction`: **TRUNCATED**, finish `length`; native text "YES\nYES\nYES\nYES\nYES\nYES\nYES\nYES\nYES\nYES\nYES". Reference text "YES\nYES\nYES\nYES\nYES\nYES\nYES\nYES\nYES\nYES\nYES". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"John\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"John\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `bounded-retrieval`: **FAIL_RETRIEVAL**, finish `stop`; native text "The secret value is **Orchid-17**.". Reference text "The secret value is **Orchid-17**.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M05 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | ed59fe219a834ae7 | 256 | 2 | stop | PASS | PASS |
| instruction | 8fd39ab149922d90 | 32 | 1 | stop | PASS | PASS |
| json | 2d6db1603b62aee3 | 128 | 23 | stop | FAIL_JSON | FAIL_JSON |
| explanation | ab53b16a1964bb72 | 128 | 25 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 23a0cf4c6f981c8e | 64 | 5 | stop | PASS | PASS |
| multiturn | b54d20bd41372475 | 64 | 1 | stop | PASS | PASS |

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```\n". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M05 primary (affine_q4)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | ed59fe219a834ae7 | 256 | 2 | stop | FAIL_ARITHMETIC | FAIL_ARITHMETIC |
| instruction | 8fd39ab149922d90 | 32 | 1 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| json | 2d6db1603b62aee3 | 128 | 23 | stop | FAIL_JSON | FAIL_JSON |
| explanation | ab53b16a1964bb72 | 128 | 24 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 23a0cf4c6f981c8e | 64 | 5 | stop | PASS | PASS |
| multiturn | b54d20bd41372475 | 64 | 1 | stop | PASS | PASS |

Failure `arithmetic`: **FAIL_ARITHMETIC**, finish `stop`; native text "50". Reference text "50\n". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `instruction`: **FAIL_INSTRUCTION**, finish `stop`; native text "Yes". Reference text "Yes\n". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```\n". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M06 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 2 | stop | PASS | PASS |
| arithmetic | 6143384c3916f109 | 256 | 10 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| instruction | de9e8994e7989173 | 32 | 2 | stop | PASS | PASS |
| json | 92c177aecf26ef40 | 128 | 16 | stop | PASS | PASS |
| explanation | 418bb592a5b9f989 | 128 | 32 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 54a1098b97de114f | 64 | 11 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |
| multiturn | a7a60b3531e74bcb | 64 | 2 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |

Failure `arithmetic`: **FAIL_INSTRUCTION**, finish `stop`; native text "17 + 25 = 42". Reference text "17 + 25 = 42". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `bounded-retrieval`: **FAIL_INSTRUCTION**, finish `stop`; native text "The secret value is \"orchid-17\".". Reference text "The secret value is \"orchid-17\".". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `multiturn`: **FAIL_INSTRUCTION**, finish `stop`; native text "Lin.". Reference text "Lin.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M06 primary (affine_q4)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 18 | stop | PASS | PASS |
| arithmetic | 6143384c3916f109 | 256 | 2 | stop | FAIL_ARITHMETIC | FAIL_ARITHMETIC |
| instruction | de9e8994e7989173 | 32 | 2 | stop | PASS | PASS |
| json | 92c177aecf26ef40 | 128 | 16 | stop | PASS | PASS |
| explanation | 418bb592a5b9f989 | 128 | 43 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 54a1098b97de114f | 64 | 1 | stop | FAIL_RETRIEVAL | FAIL_RETRIEVAL |
| multiturn | a7a60b3531e74bcb | 64 | 2 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |

Failure `arithmetic`: **FAIL_ARITHMETIC**, finish `stop`; native text "32". Reference text "32". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `bounded-retrieval`: **FAIL_RETRIEVAL**, finish `stop`; native text "background". Reference text "background". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `multiturn`: **FAIL_INSTRUCTION**, finish `stop`; native text "Lin.". Reference text "Lin.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M07 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | c7fcb51cd8271c5d | 256 | 1 | stop | PASS | PASS |
| instruction | ee4f0285d6e936d1 | 32 | 1 | stop | PASS | PASS |
| json | e736f81407a089c2 | 128 | 57 | stop | FAIL_JSON | TRUNCATED |
| explanation | fbc90d674ca88517 | 128 | 34 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | ce43219fc7db5df5 | 64 | 4 | stop | PASS | PASS |
| multiturn | e4c32dea251f3ddc | 64 | 1 | stop | PASS | PASS |

Failure `json`: **FAIL_JSON**, finish `stop`; native text "Here is a simple JSON object with the specified keys:\n\n```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```\n\nThis JSON object has exactly two keys: \"name\" with value \"Lin\" and \"count\" with value 2.". Reference text "Here's a simple Python function that returns a JSON object with the specified keys:\n\n```python\nimport json\n\ndef create_json_object():\n    \"\"\"\n    Returns a JSON object with 'name' and 'count' keys.\n    The 'name' key has a value of 'Lin', and the 'count' key has a value of 2.\n    \"\"\"\n    return {\n        \"name\": \"Lin\",\n        \"count\": 2\n    }\n\n# Example usage:\njson_object = create_json_object()\nprint(json.dumps(json_object, indent=4))\n```\n\nWhen you run this code, it will output:\n\n```json\n{\n". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M07 primary (affine_q4)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | c7fcb51cd8271c5d | 256 | 1 | stop | PASS | PASS |
| instruction | ee4f0285d6e936d1 | 32 | 1 | stop | PASS | PASS |
| json | e736f81407a089c2 | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | fbc90d674ca88517 | 128 | 40 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | ce43219fc7db5df5 | 64 | 4 | stop | PASS | PASS |
| multiturn | e4c32dea251f3ddc | 64 | 1 | stop | PASS | PASS |

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M08 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | 43b7d959a370bc0e | 256 | 2 | stop | PASS | PASS |
| instruction | f65fd7843ce3d62a | 32 | 1 | stop | PASS | PASS |
| json | f08df661bdcc804b | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 68cfecef6ab358d5 | 128 | 32 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | f5a2897003a7a5d1 | 64 | 5 | stop | PASS | PASS |
| multiturn | ec51a550ea046748 | 64 | 1 | stop | PASS | PASS |

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M08 primary (affine_q4)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | 43b7d959a370bc0e | 256 | 2 | stop | PASS | PASS |
| instruction | f65fd7843ce3d62a | 32 | 1 | stop | PASS | PASS |
| json | f08df661bdcc804b | 128 | 20 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 68cfecef6ab358d5 | 128 | 25 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | f5a2897003a7a5d1 | 64 | 1 | stop | FAIL_RETRIEVAL | FAIL_RETRIEVAL |
| multiturn | ec51a550ea046748 | 64 | 2 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `bounded-retrieval`: **FAIL_RETRIEVAL**, finish `stop`; native text " background". Reference text " background". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `multiturn`: **FAIL_INSTRUCTION**, finish `stop`; native text "Lin.". Reference text "Lin.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M09 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | f1c0337b4d9013d4 | 256 | 3 | stop | PASS | PASS |
| instruction | 31d404c55e4c16c9 | 32 | 2 | stop | PASS | PASS |
| json | 9c2178c61b503221 | 128 | 24 | stop | FAIL_JSON | FAIL_JSON |
| explanation | b0d0107cca954faf | 128 | 28 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 5c29b571be9c2e35 | 64 | 5 | stop | PASS | PASS |
| multiturn | eeb3fcc119935983 | 64 | 1 | stop | PASS | PASS |

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```\n". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```\n". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M10 stored (stored)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 16 | stop | PASS | PASS |
| arithmetic | 6742c5c0bc63b768 | 256 | 4 | stop | PASS | PASS |
| instruction | 50c2bb9b90895cac | 32 | 2 | stop | PASS | PASS |
| json | 4214851f71322be8 | 128 | 24 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 82f6f4fa83253ccc | 128 | 30 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 1903e5851406df3d | 64 | 6 | stop | PASS | PASS |
| multiturn | 50b64574c277f99b | 64 | 3 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `multiturn`: **FAIL_INSTRUCTION**, finish `stop`; native text "Lin.". Reference text "Lin.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

## M10 primary (affine_q4)

Template present: `True`. [Exact rendered prompts and generated token IDs](../archived-evidence.md).

| Case | Prompt SHA256 prefix | Budget | Generated IDs count | Finish | Native | Reference |
|---|---|---:|---:|---|---|---|
| completion | bbaff4d2ecd5892d | 32 | 32 | length | PASS | PASS |
| arithmetic | 6742c5c0bc63b768 | 256 | 4 | stop | PASS | PASS |
| instruction | 50c2bb9b90895cac | 32 | 32 | length | TRUNCATED | TRUNCATED |
| json | 4214851f71322be8 | 128 | 24 | stop | FAIL_JSON | FAIL_JSON |
| explanation | 82f6f4fa83253ccc | 128 | 28 | stop | PASS | PASS |
| repetition | 595fefd16d78948a | 32 | 32 | length | PASS | PASS |
| bounded-retrieval | 1903e5851406df3d | 64 | 6 | stop | PASS | PASS |
| multiturn | 50b64574c277f99b | 64 | 3 | stop | FAIL_INSTRUCTION | FAIL_INSTRUCTION |

Failure `instruction`: **TRUNCATED**, finish `length`; native text "NO As a historian specializing in the study of ancient civilizations, craft a detailed analysis of the political structure of the Sumerian city-state". Reference text "NO As a historian specializing in the study of ancient civilizations, craft a detailed analysis of the political structure of the Sumerian city-state". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `json`: **FAIL_JSON**, finish `stop`; native text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". Reference text "```json\n{\n  \"name\": \"Lin\",\n  \"count\": 2\n}\n```". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

Failure `multiturn`: **FAIL_INSTRUCTION**, finish `stop`; native text "Lin.". Reference text "Lin.". No prompt, expected answer or difficult case was removed. Token IDs are retained verbatim in the linked record.

All 40 historical task failures were individually reviewed against their preserved outputs, token IDs and finish reasons in [legacy-quality-failure-audit.json](../archived-evidence.md). Correct arithmetic with extra expression text remains a failed exact-format instruction, distinct from a wrong numerical answer. Source M02 and SmolLM2 produce 42 with extra text, whereas their primary quantized variants produce 32; identical task scores therefore hide arithmetic degradation. Source SmolLM2 retrieves the correct literal value with framing, while Q4 returns background. No failure classification change promotes a failed response to PASS.

Historical 32-token records remain in the previous cohort. A failed response ending in EOS/stop is an instruction/arithmetic/JSON/retrieval failure under this exact task contract; extra budget does not retroactively turn it into truncation. A failed response ending at its new token limit is TRUNCATED. A valid exact response or valid expected JSON can pass at a length boundary. No invalid JSON is accepted. Source versus primary scores above are executed separately; they are not inferred from greedy token parity.
