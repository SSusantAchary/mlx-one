# Precision status — 2026-10-20

| Target | Existing implementation | New measured scope | Qualification |
|---|---|---|---|
| FP32 | Floating safetensors and native modules; preserve stored precision with execution_dtype=None | GPT-2 exact checkpoint, B1 16/16 forced decode; raw HTTP/Python smoke | Smoke passes; numerical gate FAIL |
| FP16 | Explicit native floating execution policy; packed payload remains integer | Qwen2/Qwen3 Q8 and Qwen3.5 Q4 derivatives | Numerical gates FAIL |
| BF16 | Safetensors load natively, no NumPy weight conversion; explicit BF16 policy exists | LFM2 and SmolLM2 storage inspected; as-stored logits observed FP32 | BF16 compute qualification NOT_RUN |
| Affine Q4 | Packed uint32 modules, group sizes validated, scale/bias shape/key checks | Qwen3.5 derivative with FP16 activations; norm-mapping fix | Numerical gate FAIL |
| Affine Q8 | Packed uint32 modules retained; explicit activation dtype independent of storage | Qwen2/Qwen3 derivatives with FP16 activations | Numerical gates FAIL |

A registration, dtype declaration, quantized weight load or matching greedy output never certifies a checkpoint/precision/task cell. No packed Q4/Q8 payload was dequantized and relabeled INT8. Observed weight dtypes are recorded in each source-bound report. Core describe_architecture reports declarations and NOT_QUALIFIED; existing describe_precision remains authoritative for storage/compute distinction. Full FP32/FP16/BF16/Q4/Q8 checkpoint matrix, quantized full-model parity, memory safety, context and endurance gates remain open. No precision profile is newly release-qualified.
