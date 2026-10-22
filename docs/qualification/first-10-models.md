# First ten checkpoints — 2026-10-20

Architecture progress, exact checkpoint availability, and numerical/API qualification are distinct. All required model sources exist in both read-only snapshots. No internet source retrieval or checkpoint download was used. See [offline inventory](archived-evidence.md) for exact local revisions, paths, asset sizes, source/config hashes and separately named alternatives.

| Order | Requested checkpoint | Native family | Exact assets | Result / blocker | Release status |
|---|---|---|---|---|---|
| 1 | `openai-community/gpt2` | Registered | Local | FP32: 16/18 last-token logit boundaries pass; all 16 forced greedy choices agree; full/cached final passes. Python/HTTP/SSE smoke PASS. Numerical gate FAIL. | NOT_QUALIFIED |
| 2 | `Qwen/Qwen2.5-0.5B-Instruct` | Registered | Missing | Exact checkpoint absent. Q8 Instruct derivative diagnostic: 0/18 numerical boundaries pass, 16/16 forced greedy choices agree. Base 0.5B also cached; not an Instruct substitute. | NOT_QUALIFIED |
| 3 | `Qwen/Qwen3-0.6B` | Registered | Missing | Exact checkpoint absent. Q8 derivative diagnostic: 0/18 numerical boundaries pass, 16/16 forced greedy choices agree. Historical FP16 layerwise evidence preserved. | NOT_QUALIFIED |
| 4 | `LiquidAI/LFM2-350M` | Registered | Local | Exact BF16 assets loaded; as-stored native logits promote to FP32. 0/18 FP32 numerical boundaries pass, 16/16 forced greedy choices agree. Final full/cached gate fails in both implementations; no defect attribution. | NOT_QUALIFIED |
| 5 | `Qwen/Qwen3.5-0.8B` | Registered | Missing | Exact checkpoint absent. Q4 derivative: corrected folded norm loading; 16/16 forced greedy choices now agree; max absolute logit error 0.0546875, unchanged FP16 gate FAIL. | NOT_QUALIFIED |
| 6 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Registered | Local | Exact BF16 assets loaded; as-stored native logits promote to FP32. 0/18 FP32 numerical boundaries pass, 16/16 forced greedy choices agree. BF16 qualification remains open. | NOT_QUALIFIED |
| 7 | `meta-llama/Llama-3.2-1B-Instruct` | Registered | Missing | Exact checkpoint absent; native Llama exists but Llama3 frequency scaling is unsupported. No checkpoint run. | NOT_QUALIFIED |
| 8 | `Qwen/Qwen2.5-1.5B-Instruct` | Registered | Missing | Exact checkpoint absent. Base Qwen/Qwen2.5-1.5B assets cached, not interchangeable with Instruct. No new checkpoint run. | NOT_QUALIFIED |
| 9 | `google/gemma-3-1b-it` | Absent | Missing | Both local reference architectures present. Native Gemma3 absent; exact checkpoint absent. No implementation/qualification claim. | NOT_QUALIFIED |
| 10 | `microsoft/Phi-3-mini-4k-instruct` | Absent | Missing | Both local reference architectures present. Native Phi3 absent; exact checkpoint absent. No implementation/qualification claim. | NOT_QUALIFIED |

The six executed parity profiles are B1, 16 input tokens, 16 forced continuation steps. They are diagnostics, not free-generation, quality, full-vocabulary-at-every-input-position, HTTP load or performance qualification. Greedy agreement means argmax at each shared reference continuation; no EOS is applied in that forced track. All reports pin checkpoint revision and weight hashes. Original BF16 profiles intentionally retain their observed FP32 logits and a declared FP32 comparison gate; their failures do not qualify BF16 or establish an architectural defect.

Qwen3.5 remains at `models/vision_language/qwen3_5/` to preserve existing multimodal imports and real convolution/recurrent state. Llama and SmolLM2 retain the shared Llama implementation. No dense approximation or fake Gemma3/Phi3 registration was added while prerequisite mathematical qualification gates remain open.
