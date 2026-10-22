> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Oracle self-consistency

Cohort: `numerical-correction-20261020-1792496951619201000`. Frozen gates remain unchanged. The A/B/C/D paths execute on identical explicitly recorded IDs and source configuration. A/B call each model directly with `cache=None` for every prefix; C/D use independent fresh caches and process one token at a time. Arrays are evaluated and copied into NumPy before the next state update. No model wrapper silently retains full-path cache state. For these additional pairwise diagnostics, the first named path is the numerical reference in the frozen abs+relative equation; this direction is fixed by PAIRS before execution. The established central MLX-LM-to-native numerical contract is unchanged and remains independently required. Native and reference are in evaluation mode.

Each source and primary track tests every prefix through 64 tokens, with requested partition lengths 1, 2, 3, 7, 15, 16, 17, 32 and 64. Partitions include [N], [N-1,1], [N-2,2], [1,N-1], groups of 4 and groups of 8, where valid. Tables give the first failing one-based prefix length, not a zero-based position. A source-precision result does not qualify Q4/Q8.

A = native fresh prefix; B = local reference fresh prefix; C = native cache; D = reference cache. All six pairings compare the complete vocabulary. Chunk checks compare every logical token to its independently recomputed prefix. Token inputs are tokenizer IDs without an extra BOS. Position offsets begin at zero; causal masks follow the actual query and key widths. Actual dtypes and offsets are in each evidence record. A null recurrent reference offset means ArraysCache has no offset attribute; the explicit partition cursor tracks tokens and KV layers carry RoPE offsets.

## Stored execution

| ID | Precision | A/B | C/D | A/C | B/D | A/D | B/C | Failed chunks native/ref | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| M01 | stored | FAIL @20 | FAIL @20 | FAIL @20 | FAIL @20 | FAIL @24 | FAIL @20 | 6/9 | [M01.json](../archived-evidence.md) |
| M02 | stored | FAIL @1 | FAIL @1 | FAIL @2 | FAIL @2 | FAIL @1 | FAIL @1 | 40/40 | [M02.json](../archived-evidence.md) |
| M03 | stored | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 40/40 | [M03.json](../archived-evidence.md) |
| M04 | stored | FAIL @5 | FAIL @3 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 40/40 | [M04.json](../archived-evidence.md) |
| M05 | stored | FAIL @1 | FAIL @1 | FAIL @2 | FAIL @2 | FAIL @1 | FAIL @1 | 40/40 | [M05.json](../archived-evidence.md) |
| M06 | stored | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 40/40 | [M06.json](../archived-evidence.md) |
| M07 | stored | FAIL @2 | FAIL @4 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 38/38 | [M07.json](../archived-evidence.md) |
| M08 | stored | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 40/40 | [M08.json](../archived-evidence.md) |
| M09 | stored | PASS | PASS | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 40/40 | [M09.json](../archived-evidence.md) |
| M10 | stored | PASS | PASS | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 39/39 | [M10.json](../archived-evidence.md) |

## Primary execution

| ID | Precision | A/B | C/D | A/C | B/D | A/D | B/C | Failed chunks native/ref | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| M01 | stored | FAIL @20 | FAIL @20 | FAIL @20 | FAIL @20 | FAIL @24 | FAIL @20 | 6/9 | [M01.json](../archived-evidence.md) |
| M02 | affine_q8 | FAIL @2 | FAIL @2 | PASS | FAIL @2 | FAIL @2 | FAIL @2 | 0/40 | [M02.json](../archived-evidence.md) |
| M03 | affine_q8 | FAIL @2 | FAIL @2 | PASS | FAIL @2 | FAIL @2 | FAIL @2 | 0/35 | [M03.json](../archived-evidence.md) |
| M04 | affine_q4 | FAIL @2 | FAIL @2 | PASS | FAIL @2 | FAIL @2 | FAIL @2 | 0/40 | [M04.json](../archived-evidence.md) |
| M05 | affine_q4 | FAIL @1 | FAIL @1 | PASS | FAIL @2 | FAIL @1 | FAIL @1 | 0/40 | [M05.json](../archived-evidence.md) |
| M06 | affine_q4 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 20/40 | [M06.json](../archived-evidence.md) |
| M07 | affine_q4 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 20/40 | [M07.json](../archived-evidence.md) |
| M08 | affine_q4 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 35/40 | [M08.json](../archived-evidence.md) |
| M09 | stored | PASS | PASS | FAIL @2 | FAIL @2 | FAIL @2 | FAIL @2 | 40/40 | [M09.json](../archived-evidence.md) |
| M10 | affine_q4 | PASS | PASS | PASS | PASS | PASS | PASS | 32/32 | [M10.json](../archived-evidence.md) |

The all-parameter comparison checks every shared text-forward array in all 18 source/primary profiles. There are no unexplained mapped differences. Singleton convolution storage axes and Qwen3.5 BF16-folded offset norms are mapped explicitly; norm representation equivalence does not assert identical runtime norm arithmetic. Qwen3.5 native-only vision and MTP side-branch arrays have no local text-only comparator and remain outside this numerical claim. Pinned checkpoint-file hashes match the prior cohort. See [parameter-integrity-summary.json](../archived-evidence.md) and [checkpoint-file-integrity.json](../archived-evidence.md).

## Gemma3 and Phi3 adjudication

The historical discrepancies were Gemma3 decode/chunk maxima 0.5/1.5625 and Phi3 0.25/0.5. Both reproduced exactly before implementation changes. The new untraced oracle finds shared full/cache violations at prefix length 2, while A/B and C/D are bitwise equal through 64 source-weight tokens. This rules out a native-only cache failure for these workloads; it does not qualify the shared numerical behavior.

At token 2, Gemma3 first differs at layer 1 MLP up_proj (one BF16 element, 2.384185791015625e-7); Phi3 first differs at layer 0 qkv_proj (one BF16 element, 1.52587890625e-5). The inputs are exactly equal. Replaying a full projection and its final row with identical weights reproduces each discrepancy. Both projection results pass against an independent CPU float64 dot product. The first traced self-consistency threshold violations are Gemma3 layer 3 post_feedforward_layernorm and Phi3 layer 17 output. These traces are diagnostics; the untraced paths determine gate verdicts.

M09: a diagnostic replay using fixed FP32 row reductions for linears and explicit attention gives **PASS** across all six pairs and partitions at lengths [1, 2, 3, 7, 16]. It preserves output dtype and changes no cache semantics. `qualification_eligible` is false: these substituted reference/native operators are not the production profile. [Diagnostic replay](../archived-evidence.md). The combined replay demonstrates a numerical-order mechanism; it is not a promotion of a different backend or proof of the accuracy of all production boundaries.

M10: a diagnostic replay using fixed FP32 row reductions for linears and explicit attention gives **PASS** across all six pairs and partitions at lengths [1, 2, 3, 7, 16]. It preserves output dtype and changes no cache semantics. `qualification_eligible` is false: these substituted reference/native operators are not the production profile. [Diagnostic replay](../archived-evidence.md). The combined replay demonstrates a numerical-order mechanism; it is not a promotion of a different backend or proof of the accuracy of all production boundaries.

Exact-weight attention-layer sliding-window controls pass at Gemma3 lengths 511/512/513 (local layer 0 and global layer 5), and Phi3 lengths 2046/2047/2048. These test layer invariants, not whole-model long-context qualification. The local Phi3 model omits configured sliding masks; its attention layer receives an explicit source-derived mask in the control. The reference source is unchanged.

The first Gemma boundary probe used a pre-eviction native K width for the reference rotating cache and failed with a broadcast error. The corrected probe uses the reference returned K width and absolute position. The first LFM2/Qwen3.5 probe also assumed an offset property on reference recurrent ArraysCache; that metadata error is fixed. Failed probes remain archived and contribute no PASS verdict. Regression tests cover both contracts.

Phi3 needs a second controlled distinction: fixed-row reductions alone leave a reference-only failure starting at prefix 5, first traced in layer 9 attention output and violating at layer 22. Making Q/K/V contiguous in the same diagnostic arithmetic removes that failure. The 16-token contiguous-rowwise control and Gemma3 16-token rowwise control both pass all six pairs and chunk partitions without changing cache semantics; both are explicitly nonqualifying. The unmodified production operator/precision profile still fails.

Canonical self-consistency remains unresolved under the frozen precision gates. No tolerance, stored precision, packed policy, or production attention backend was changed to obtain a PASS. PyTorch was not installed; Transformers was inspected locally rather than executed. CPU float64 controls independently validate selected equations and state transitions, not all ten full networks.
