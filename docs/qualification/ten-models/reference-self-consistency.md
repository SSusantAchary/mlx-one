> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Reference self-consistency

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

Path A is native fresh full prefix; B is original reference fresh full prefix; C is native incremental; D is reference incremental. All arrays are evaluated and independently copied before mutation. The first path in each added pair supplies the relative-error denominator; the original 17-boundary gates separately keep reference-to-native direction.

| ID | Reference full/cache max error | First failure prefix | Native first nonzero operation at first differing prefix | Independent equal-input projection max error | CPU full / single |
|---|---:|---:|---|---:|---|
| M01 | 0.007431983948 | 20 | `transformer.h.0.attn.c_attn#1` (prefix 2) | 3.814697266e-06 | PASS / PASS |
| M02 | 0.5859375 | 2 | `model.layers.0.self_attn.q_proj#1` (prefix 2) | 0.0625 | PASS / PASS |
| M03 | 0.375 | 2 | `model.layers.0.mlp.gate_proj#1` (prefix 2) | 3.051757812e-05 | PASS / PASS |
| M04 | 0.34375 | 2 | `model.layers.0.conv.out_proj#1` (prefix 2) | 2.980232239e-08 | PASS / PASS |
| M05 | 0.28125 | 2 | `language_model.layers.0.linear_attn.in_proj_qkv#1` (prefix 2) | 9.403954807e-38 | PASS / PASS |
| M06 | 0.4375 | 2 | `model.layers.0.self_attn.v_proj#1` (prefix 2) | 0.000244140625 | PASS / PASS |
| M07 | 0.2421875 | 2 | `model.layers.0.self_attn.o_proj#1` (prefix 2) | 2.384185791e-07 | PASS / PASS |
| M08 | 0.65625 | 2 | `model.layers.0.self_attn.q_proj#1` (prefix 2) | 0.0625 | PASS / PASS |
| M09 | 1.5 | 2 | `model.layers.1.mlp.up_proj#1` (prefix 2) | 2.384185791e-07 | PASS / PASS |
| M10 | 0.5 | 2 | `model.layers.0.self_attn.qkv_proj#1` (prefix 2) | 1.525878906e-05 | PASS / PASS |

All ten equal-input probes have exactly equal full/cache inputs at the selected projection. Both reduction shapes pass the independent FP64 component tolerance; their differences can amplify downstream. Replacing projection and attention reductions with bounded per-row FP32 controls makes native and reference own full/cache and chunk paths exactly equal through 16 for all ten. All six cross pairings pass for nine; Qwen3.5 still differs across implementations (max 0.25). These are diagnostic substitutions with qualification_eligible=false, not production backends or release passes. These interventions establish reduction-path sensitivity; the first small nonzero value alone is not asserted to explain all later failures.

The four-dimensional trace selector now distinguishes normalization before head transpose from [B,H,T,D] outputs. Eight independent CPU tests pin the token axis. Existing Gemma3/Phi3 logit and trace evidence was unaffected; new per-family traces use the corrected selector. [Trace evidence](../archived-evidence.md).
