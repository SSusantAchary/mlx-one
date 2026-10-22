# P0 Engine test matrix

Measured source: `53404467afa3cf094bd221c61f5316f2b928b161`; working-source SHA256 `554ce34479cf89daf094caf9843c84a2b77eb1d45968e227fafd7d91f1ea68f6`.

Local evidence: `qualification/results/p0_engine/runs/1792201272601288000`. Checkpoints, raw traces and results are uncommitted.

| Model | Case | Requirement | Suite | Mandatory/applicable | Status | Reason |
|---|---|---|---|---|---|---|
| openai-community/gpt2 | A01 | Locate pinned local snapshot. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A02 | Validate checkpoint files. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A03 | Load through mlx_one.Engine. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A04 | Verify architecture. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A05 | Verify expected weight precision. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A06 | Verify native reference_dense mode. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A07 | Verify tokenizer loads. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A08 | Verify model forward execution. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A09 | Generate at least 16 tokens. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A10 | Confirm no NaN/Inf logits. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A11 | Read Engine.capabilities(). | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A12 | Read Engine.runtime_stats(). | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A13 | Verify model identity and revision. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | A14 | Close Engine cleanly. | startup | True/True | PASS | not executed |
| openai-community/gpt2 | B01 | Five-repeat greedy token determinism across 12 corpus cases | correctness | True/True | PASS | not executed |
| openai-community/gpt2 | B02 | Five-repeat seeded sampling | correctness | True/True | PASS | not executed |
| openai-community/gpt2 | B03 | Output token limits | correctness | True/True | PASS | not executed |
| openai-community/gpt2 | B04 | EOS and overlapping/cross-token stops | correctness | True/True | PASS | not executed |
| openai-community/gpt2 | B05 | Legal rendered inputs and context overflow | correctness | True/True | PASS | not executed |
| openai-community/gpt2 | B06 | Model-specific text/chat templates | correctness | True/False | NOT_SUPPORTED | GPT-2 completion does not require chat formatting |
| openai-community/gpt2 | C01 | Valid incremental chunks. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C02 | Correct generated token order. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C03 | No duplicated token-bearing events. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C04 | No lost generated tokens. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C05 | Concatenated text matches non-streaming text. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C06 | Valid UTF-8/Unicode output. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C07 | Correct max_tokens. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C08 | Correct EOS behavior. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C09 | Correct final finish reason. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | C10 | Early stream close releases resources. | streaming | True/True | PASS | not executed |
| openai-community/gpt2 | D01 | Full fresh prefill. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D02 | Prefill + one-token decode. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D03 | Prefill + multiple decode tokens. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D04 | Fresh full-prefill versus cached continuation. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D05 | Position ID correctness. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D06 | Attention-mask correctness. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D07 | Repeat with reset model/cache state. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D08 | Divergent sequences after a common prefix. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D09 | Independent cache reconstruction. | state | True/True | PASS | not executed |
| openai-community/gpt2 | D10 | Model-specific state consistency. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E01 | Attention KV dtype is FP16. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E02 | Cache positions are correct. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E03 | Cache lengths advance correctly. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E04 | Cached decoding agrees with fresh execution. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E05 | Cache state resets correctly. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E06 | Cache state is isolated between requests. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E07 | Cache eviction leaves no invalid ownership. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E08 | Cancellation leaves no orphaned reservations. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E09 | Model reload invalidates stale state. | state | True/True | PASS | not executed |
| openai-community/gpt2 | E10 | Incorrect model/cache identity prevents reuse. | state | True/True | PASS | not executed |
| openai-community/gpt2 | F01 | Sequential submissions — 20 requests. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F02 | Concurrent submissions — 2 requests. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F03 | Concurrent submissions — 4 requests. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F04 | Concurrent submissions — 8 requests. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F05 | Queue cancellation. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F06 | Cancellation during prefill. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F07 | Cancellation during decode. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F08 | Async exception propagation. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F09 | Close with active request. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F10 | New Engine after previous Engine closes. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F11 | Async task cancellation signals its native request | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | F12 | Close an already-open direct stream | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G01 | Load model. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G02 | Generate. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G03 | Stream. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G04 | Cancel a request. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G05 | Close Engine. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G06 | Reload model. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G07 | Generate identical deterministic output. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G08 | Release all active request resources. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G09 | Repeat lifecycle testing. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | G10 | Confirm no stale state reuse. | lifecycle | True/True | PASS | not executed |
| openai-community/gpt2 | H128 | benchmark | benchmark | True/True | PASS | not executed |
| openai-community/gpt2 | H512 | benchmark | benchmark | True/True | PASS | not executed |
| openai-community/gpt2 | H2048 | benchmark | benchmark | True/False | NOT_SUPPORTED | prompt plus output exceeds checkpoint context |
| openai-community/gpt2 | H8192 | benchmark | benchmark | True/False | NOT_SUPPORTED | prompt plus output exceeds checkpoint context |
| openai-community/gpt2 | I01 | Cold exact 8192-token request | admission | True/False | NOT_SUPPORTED | 8K exceeds checkpoint context |
| openai-community/gpt2 | I02 | 8192 after shorter warmed workloads | admission | True/False | NOT_SUPPORTED | 8K exceeds checkpoint context |
| openai-community/gpt2 | I03 | 8192 after native decode cancellation | admission | True/False | NOT_SUPPORTED | 8K exceeds checkpoint context |
| openai-community/gpt2 | I04 | 8192 after explicit safe owner cleanup | admission | True/False | NOT_SUPPORTED | 8K exceeds checkpoint context |
| openai-community/gpt2 | I05 | Consecutive 8192 requests | admission | True/False | NOT_SUPPORTED | 8K exceeds checkpoint context |
| openai-community/gpt2 | I06 | Independent fresh-worker 8192 parity | admission | True/False | NOT_SUPPORTED | 8K exceeds checkpoint context |
| openai-community/gpt2 | P01 | state | state | False/False | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A01 | Locate pinned local snapshot. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A02 | Validate checkpoint files. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A03 | Load through mlx_one.Engine. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A04 | Verify architecture. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A05 | Verify expected weight precision. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A06 | Verify native reference_dense mode. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A07 | Verify tokenizer loads. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A08 | Verify model forward execution. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A09 | Generate at least 16 tokens. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A10 | Confirm no NaN/Inf logits. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A11 | Read Engine.capabilities(). | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A12 | Read Engine.runtime_stats(). | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A13 | Verify model identity and revision. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | A14 | Close Engine cleanly. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | B01 | Five-repeat greedy token determinism across 12 corpus cases | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | B02 | Five-repeat seeded sampling | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | B03 | Output token limits | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | B04 | EOS and overlapping/cross-token stops | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | B05 | Legal rendered inputs and context overflow | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | B06 | Model-specific text/chat templates | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C01 | Valid incremental chunks. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C02 | Correct generated token order. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C03 | No duplicated token-bearing events. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C04 | No lost generated tokens. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C05 | Concatenated text matches non-streaming text. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C06 | Valid UTF-8/Unicode output. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C07 | Correct max_tokens. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C08 | Correct EOS behavior. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C09 | Correct final finish reason. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | C10 | Early stream close releases resources. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D01 | Full fresh prefill. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D02 | Prefill + one-token decode. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D03 | Prefill + multiple decode tokens. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D04 | Fresh full-prefill versus cached continuation. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D05 | Position ID correctness. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D06 | Attention-mask correctness. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D07 | Repeat with reset model/cache state. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D08 | Divergent sequences after a common prefix. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D09 | Independent cache reconstruction. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | D10 | Model-specific state consistency. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E01 | Attention KV dtype is FP16. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E02 | Cache positions are correct. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E03 | Cache lengths advance correctly. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E04 | Cached decoding agrees with fresh execution. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E05 | Cache state resets correctly. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E06 | Cache state is isolated between requests. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E07 | Cache eviction leaves no invalid ownership. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E08 | Cancellation leaves no orphaned reservations. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E09 | Model reload invalidates stale state. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | E10 | Incorrect model/cache identity prevents reuse. | state | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F01 | Sequential submissions — 20 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F02 | Concurrent submissions — 2 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F03 | Concurrent submissions — 4 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F04 | Concurrent submissions — 8 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F05 | Queue cancellation. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F06 | Cancellation during prefill. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F07 | Cancellation during decode. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F08 | Async exception propagation. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F09 | Close with active request. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F10 | New Engine after previous Engine closes. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F11 | Async task cancellation signals its native request | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | F12 | Close an already-open direct stream | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G01 | Load model. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G02 | Generate. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G03 | Stream. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G04 | Cancel a request. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G05 | Close Engine. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G06 | Reload model. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G07 | Generate identical deterministic output. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G08 | Release all active request resources. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G09 | Repeat lifecycle testing. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | G10 | Confirm no stale state reuse. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | H128 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | H512 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | H2048 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | H8192 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | I01 | Cold exact 8192-token request | admission | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | I02 | 8192 after shorter warmed workloads | admission | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | I03 | 8192 after native decode cancellation | admission | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | I04 | 8192 after explicit safe owner cleanup | admission | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | I05 | Consecutive 8192 requests | admission | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | I06 | Independent fresh-worker 8192 parity | admission | True/True | PASS | not executed |
| mlx-community/Qwen3-0.6B-8bit | P01 | state | state | False/False | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A01 | Locate pinned local snapshot. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A02 | Validate checkpoint files. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A03 | Load through mlx_one.Engine. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A04 | Verify architecture. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A05 | Verify expected weight precision. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A06 | Verify native reference_dense mode. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A07 | Verify tokenizer loads. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A08 | Verify model forward execution. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A09 | Generate at least 16 tokens. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A10 | Confirm no NaN/Inf logits. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A11 | Read Engine.capabilities(). | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A12 | Read Engine.runtime_stats(). | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A13 | Verify model identity and revision. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | A14 | Close Engine cleanly. | startup | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | B01 | Five-repeat greedy token determinism across 12 corpus cases | correctness | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | B02 | Five-repeat seeded sampling | correctness | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | B03 | Output token limits | correctness | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | B04 | EOS and overlapping/cross-token stops | correctness | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | B05 | Legal rendered inputs and context overflow | correctness | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | B06 | Model-specific text/chat templates | correctness | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C01 | Valid incremental chunks. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C02 | Correct generated token order. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C03 | No duplicated token-bearing events. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C04 | No lost generated tokens. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C05 | Concatenated text matches non-streaming text. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C06 | Valid UTF-8/Unicode output. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C07 | Correct max_tokens. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C08 | Correct EOS behavior. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C09 | Correct final finish reason. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | C10 | Early stream close releases resources. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D01 | Full fresh prefill. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D02 | Prefill + one-token decode. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D03 | Prefill + multiple decode tokens. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D04 | Fresh full-prefill versus cached continuation. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D05 | Position ID correctness. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D06 | Attention-mask correctness. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D07 | Repeat with reset model/cache state. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D08 | Divergent sequences after a common prefix. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D09 | Independent cache reconstruction. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | D10 | Model-specific state consistency. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E01 | Attention KV dtype is FP16. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E02 | Cache positions are correct. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E03 | Cache lengths advance correctly. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E04 | Cached decoding agrees with fresh execution. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E05 | Cache state resets correctly. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E06 | Cache state is isolated between requests. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E07 | Cache eviction leaves no invalid ownership. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E08 | Cancellation leaves no orphaned reservations. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E09 | Model reload invalidates stale state. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | E10 | Incorrect model/cache identity prevents reuse. | state | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F01 | Sequential submissions — 20 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F02 | Concurrent submissions — 2 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F03 | Concurrent submissions — 4 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F04 | Concurrent submissions — 8 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F05 | Queue cancellation. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F06 | Cancellation during prefill. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F07 | Cancellation during decode. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F08 | Async exception propagation. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F09 | Close with active request. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F10 | New Engine after previous Engine closes. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F11 | Async task cancellation signals its native request | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | F12 | Close an already-open direct stream | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G01 | Load model. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G02 | Generate. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G03 | Stream. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G04 | Cancel a request. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G05 | Close Engine. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G06 | Reload model. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G07 | Generate identical deterministic output. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G08 | Release all active request resources. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G09 | Repeat lifecycle testing. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | G10 | Confirm no stale state reuse. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | H128 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | H512 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | H2048 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | H8192 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | I01 | Cold exact 8192-token request | admission | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | I02 | 8192 after shorter warmed workloads | admission | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | I03 | 8192 after native decode cancellation | admission | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | I04 | 8192 after explicit safe owner cleanup | admission | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | I05 | Consecutive 8192 requests | admission | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | I06 | Independent fresh-worker 8192 parity | admission | True/True | PASS | not executed |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | P01 | state | state | False/False | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A01 | Locate pinned local snapshot. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A02 | Validate checkpoint files. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A03 | Load through mlx_one.Engine. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A04 | Verify architecture. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A05 | Verify expected weight precision. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A06 | Verify native reference_dense mode. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A07 | Verify tokenizer loads. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A08 | Verify model forward execution. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A09 | Generate at least 16 tokens. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A10 | Confirm no NaN/Inf logits. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A11 | Read Engine.capabilities(). | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A12 | Read Engine.runtime_stats(). | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A13 | Verify model identity and revision. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | A14 | Close Engine cleanly. | startup | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | B01 | Five-repeat greedy token determinism across 12 corpus cases | correctness | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | B02 | Five-repeat seeded sampling | correctness | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | B03 | Output token limits | correctness | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | B04 | EOS and overlapping/cross-token stops | correctness | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | B05 | Legal rendered inputs and context overflow | correctness | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | B06 | Model-specific text/chat templates | correctness | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C01 | Valid incremental chunks. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C02 | Correct generated token order. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C03 | No duplicated token-bearing events. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C04 | No lost generated tokens. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C05 | Concatenated text matches non-streaming text. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C06 | Valid UTF-8/Unicode output. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C07 | Correct max_tokens. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C08 | Correct EOS behavior. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C09 | Correct final finish reason. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | C10 | Early stream close releases resources. | streaming | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D01 | Full fresh prefill. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D02 | Prefill + one-token decode. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D03 | Prefill + multiple decode tokens. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D04 | Fresh full-prefill versus cached continuation. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D05 | Position ID correctness. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D06 | Attention-mask correctness. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D07 | Repeat with reset model/cache state. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D08 | Divergent sequences after a common prefix. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D09 | Independent cache reconstruction. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | D10 | Model-specific state consistency. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E01 | Attention KV dtype is FP16. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E02 | Cache positions are correct. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E03 | Cache lengths advance correctly. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E04 | Cached decoding agrees with fresh execution. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E05 | Cache state resets correctly. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E06 | Cache state is isolated between requests. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E07 | Cache eviction leaves no invalid ownership. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E08 | Cancellation leaves no orphaned reservations. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E09 | Model reload invalidates stale state. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | E10 | Incorrect model/cache identity prevents reuse. | state | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F01 | Sequential submissions — 20 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F02 | Concurrent submissions — 2 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F03 | Concurrent submissions — 4 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F04 | Concurrent submissions — 8 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F05 | Queue cancellation. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F06 | Cancellation during prefill. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F07 | Cancellation during decode. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F08 | Async exception propagation. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F09 | Close with active request. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F10 | New Engine after previous Engine closes. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F11 | Async task cancellation signals its native request | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | F12 | Close an already-open direct stream | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G01 | Load model. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G02 | Generate. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G03 | Stream. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G04 | Cancel a request. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G05 | Close Engine. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G06 | Reload model. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G07 | Generate identical deterministic output. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G08 | Release all active request resources. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G09 | Repeat lifecycle testing. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | G10 | Confirm no stale state reuse. | lifecycle | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | H128 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | H512 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | H2048 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | H8192 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | I01 | Cold exact 8192-token request | admission | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | I02 | 8192 after shorter warmed workloads | admission | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | I03 | 8192 after native decode cancellation | admission | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | I04 | 8192 after explicit safe owner cleanup | admission | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | I05 | Consecutive 8192 requests | admission | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | I06 | Independent fresh-worker 8192 parity | admission | True/True | PASS | not executed |
| mlx-community/LFM2-350M-4bit | P01 | state | state | False/False | NOT_SUPPORTED | dense hybrid prefix restoration is outside this scope |
| mlx-community/Qwen3.5-0.8B-4bit | A01 | Locate pinned local snapshot. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A02 | Validate checkpoint files. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A03 | Load through mlx_one.Engine. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A04 | Verify architecture. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A05 | Verify expected weight precision. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A06 | Verify native reference_dense mode. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A07 | Verify tokenizer loads. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A08 | Verify model forward execution. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A09 | Generate at least 16 tokens. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A10 | Confirm no NaN/Inf logits. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A11 | Read Engine.capabilities(). | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A12 | Read Engine.runtime_stats(). | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A13 | Verify model identity and revision. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | A14 | Close Engine cleanly. | startup | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | B01 | Five-repeat greedy token determinism across 12 corpus cases | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | B02 | Five-repeat seeded sampling | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | B03 | Output token limits | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | B04 | EOS and overlapping/cross-token stops | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | B05 | Legal rendered inputs and context overflow | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | B06 | Model-specific text/chat templates | correctness | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C01 | Valid incremental chunks. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C02 | Correct generated token order. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C03 | No duplicated token-bearing events. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C04 | No lost generated tokens. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C05 | Concatenated text matches non-streaming text. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C06 | Valid UTF-8/Unicode output. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C07 | Correct max_tokens. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C08 | Correct EOS behavior. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C09 | Correct final finish reason. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | C10 | Early stream close releases resources. | streaming | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D01 | Full fresh prefill. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D02 | Prefill + one-token decode. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D03 | Prefill + multiple decode tokens. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D04 | Fresh full-prefill versus cached continuation. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D05 | Position ID correctness. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D06 | Attention-mask correctness. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D07 | Repeat with reset model/cache state. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D08 | Divergent sequences after a common prefix. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D09 | Independent cache reconstruction. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | D10 | Model-specific state consistency. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E01 | Attention KV dtype is FP16. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E02 | Cache positions are correct. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E03 | Cache lengths advance correctly. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E04 | Cached decoding agrees with fresh execution. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E05 | Cache state resets correctly. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E06 | Cache state is isolated between requests. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E07 | Cache eviction leaves no invalid ownership. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E08 | Cancellation leaves no orphaned reservations. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E09 | Model reload invalidates stale state. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | E10 | Incorrect model/cache identity prevents reuse. | state | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F01 | Sequential submissions — 20 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F02 | Concurrent submissions — 2 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F03 | Concurrent submissions — 4 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F04 | Concurrent submissions — 8 requests. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F05 | Queue cancellation. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F06 | Cancellation during prefill. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F07 | Cancellation during decode. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F08 | Async exception propagation. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F09 | Close with active request. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F10 | New Engine after previous Engine closes. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F11 | Async task cancellation signals its native request | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | F12 | Close an already-open direct stream | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G01 | Load model. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G02 | Generate. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G03 | Stream. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G04 | Cancel a request. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G05 | Close Engine. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G06 | Reload model. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G07 | Generate identical deterministic output. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G08 | Release all active request resources. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G09 | Repeat lifecycle testing. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | G10 | Confirm no stale state reuse. | lifecycle | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | H128 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | H512 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | H2048 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | H8192 | benchmark | benchmark | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | I01 | Cold exact 8192-token request | admission | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | I02 | 8192 after shorter warmed workloads | admission | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | I03 | 8192 after native decode cancellation | admission | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | I04 | 8192 after explicit safe owner cleanup | admission | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | I05 | Consecutive 8192 requests | admission | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | I06 | Independent fresh-worker 8192 parity | admission | True/True | PASS | not executed |
| mlx-community/Qwen3.5-0.8B-4bit | P01 | state | state | False/False | NOT_SUPPORTED | dense hybrid prefix restoration is outside this scope |

The case ledger includes actual commands, revision, source, timestamps, artifacts and individual details in `qualification_results.json`. PASS/FAIL/SKIP/NOT_RUN/BLOCKED/NOT_SUPPORTED remain distinct. EOS control cases evaluate real weights and explicitly label forced sampler termination; stochastic output diversity is not required.
