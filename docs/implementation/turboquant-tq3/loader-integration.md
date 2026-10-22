# Native loader and runtime integration

`core/quantization.py` recognizes TurboQuant before ordinary affine parsing and returns the complete
validated metadata. `TQ3Config` rejects wrong bits/groups/codebook, QJL, fused rotations, missing
fields, unsupported rotation, noninteger seeds and malformed overrides. `core/precision.py` reports
the exact TQ3 label while leaving unobserved activation/KV dtypes unspecified.

`text/loading.py::_prepare_quantized_model` branches to `prepare_tq3_model` before `nn.quantize`.
The loader detects actual codebook-bearing modules, verifies source logical dimensions, exact
weight/scale/codebook/sign/bias shapes and dtypes, finite nonnegative scales, exact codebook values
and ±1 signs. Missing, extra or inconsistent parameters fail strict model loading. Ordinary packed
exceptions require an explicit affine/MXFP4/NVFP4 spec; floating routers/embeddings stay floating.
Existing GPT MXFP4-Q4 loading and all other formats retain their previous path.

Packed arrays are installed by reference, not unpacked, repacked or copied into dense placeholders.
The base architecture's lazy initialization buffers are replaced before evaluation. The existing
owner-thread materialization evaluates only the final compressed buffers and small constants;
it does not expand weights. This prevents lazy file-read/cast streams leaking across device-owner
threads. File finite validation remains bounded CPU scanning. Full-checkpoint memory checks remain
in the shared loader. The benchmark additionally preflights available/reclaimable host memory,
working-set limits, conservative KV/scratch estimates and context bounds.

GPT `make_cache()` configures existing floating FP16 KV storage when the TQ3 marker is set. It reuses
the existing full/sliding cache implementations; the full-cache storage class is historically named
`QuantizedKVCache` even when both bit widths are None and storage is ordinary FP16. Engine/server
ownership, cancellation, reset and request isolation remain unchanged. No TQ KV compression exists.

Python/CLI/HTTP reach these modules through the existing `gpt_oss` registry entry and
`reference_dense` engine policy. This policy name does not disable the TQ3 kernels in projections.
Use default checkpoint execution dtypes. Global affine optimized/execution-dtype conversion policies
are not TQ3 conversion policies and may reject this checkpoint explicitly. Harmony templates and
multiple EOS IDs use the existing native tokenizer/parser; only assistant final output is rendered.

Support manifest adds candidate M13 without promoting either large GPT checkpoint.
Release static validation expects 13 candidates; package audit now checks the TQ3 notice explicitly.
No dependency on MLX-LM, Transformers or TurboQuant was added to production.
