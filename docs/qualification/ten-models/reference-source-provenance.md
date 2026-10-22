> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Reference source provenance

Both local source snapshots lack .git metadata. No commit is inferred from the installed comparator or old candidate headers. Content hashes pin these sources. Transformers declares 5.21.0.dev0; MLX-LM uses a dynamic version and has no generated version file. The preexisting candidate's ed1fca4… identity is not asserted for this supplied local tree. No reference tree is modified, downloaded, added as a production dependency or packaged.

Full hashes: [reference-first-20261020-1792502347743318000/port-provenance.json](../archived-evidence.md). Exact checkpoint repositories, full revisions and every asset hash: [assets.json](../archived-evidence.md). Entry source/reference inventories and final integrity verification are retained in this cohort.

| Family | MLX-LM source | SHA256 | Transformers specification |
|---|---|---|---|
| gpt2 | `references_libs/mlx-lm-main/mlx_lm/models/gpt2.py` | `6a513397b9f959ea44064dc91df0fd25db3c6eacdee6628c444204e12a549093` | `references_libs/transformers-main/src/transformers/models/gpt2/configuration_gpt2.py`, `references_libs/transformers-main/src/transformers/models/gpt2/modeling_gpt2.py` |
| qwen2 | `references_libs/mlx-lm-main/mlx_lm/models/qwen2.py` | `937e0321ef06a34efbd66d2ce0ce1e715aab98da4acefa376c8cd29291a0bc83` | `references_libs/transformers-main/src/transformers/models/qwen2/configuration_qwen2.py`, `references_libs/transformers-main/src/transformers/models/qwen2/modeling_qwen2.py` |
| qwen3 | `references_libs/mlx-lm-main/mlx_lm/models/qwen3.py` | `7e29b7055d8e526a0ede6e2740a3038ad24d0cef6a1abb446c411d3fbd0efa2d` | `references_libs/transformers-main/src/transformers/models/qwen3/configuration_qwen3.py`, `references_libs/transformers-main/src/transformers/models/qwen3/modeling_qwen3.py` |
| lfm2 | `references_libs/mlx-lm-main/mlx_lm/models/lfm2.py` | `558dcbec61442b1f7216c0122d08d44083ea71d1510be23f4f934e62b0f71376` | `references_libs/transformers-main/src/transformers/models/lfm2/configuration_lfm2.py`, `references_libs/transformers-main/src/transformers/models/lfm2/modeling_lfm2.py` |
| qwen3_5 | `references_libs/mlx-lm-main/mlx_lm/models/qwen3_5.py` | `851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6` | `references_libs/transformers-main/src/transformers/models/qwen3_5/configuration_qwen3_5.py`, `references_libs/transformers-main/src/transformers/models/qwen3_5/modeling_qwen3_5.py` |
| llama | `references_libs/mlx-lm-main/mlx_lm/models/llama.py` | `dfb656511e6740781cd0563f7994a78b3404c199b9aebaa2fbca28a19e280891` | `references_libs/transformers-main/src/transformers/models/llama/configuration_llama.py`, `references_libs/transformers-main/src/transformers/models/llama/modeling_llama.py` |
| gemma3_text | `references_libs/mlx-lm-main/mlx_lm/models/gemma3_text.py` | `884bb398288beda5e90caf3de60a15f5d17b8e383d28c85cca07de1e0aeafa38` | `references_libs/transformers-main/src/transformers/models/gemma3/configuration_gemma3.py`, `references_libs/transformers-main/src/transformers/models/gemma3/modeling_gemma3.py` |
| phi3 | `references_libs/mlx-lm-main/mlx_lm/models/phi3.py` | `863b8af7cd7f00ec24e7196102fc83c243a712b92d09660bab870a47aff0248f` | `references_libs/transformers-main/src/transformers/models/phi3/configuration_phi3.py`, `references_libs/transformers-main/src/transformers/models/phi3/modeling_phi3.py` |

MLX-LM base mask/cache behavior, compiled activations, rotary helpers, gated_delta and qwen3_next are separately hashed in the manifest. Transformers normalization, attention/GQA, RoPE, convolution/recurrence, model configuration and mask selection are consulted directly from these local files. PyTorch execution is optional and was not used.

The MIT source copyright headers and full Apple MIT notice are preserved at `src/mlx_one/models/UPSTREAM_LICENSE`, included by wheel package-data. Existing Gemma3, Phi3 and Qwen3 notices remain. Transformers was used as specification; its Apache-2.0 code is not imported at runtime.
