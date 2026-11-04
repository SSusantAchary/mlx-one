# Exact output comparisons

For each priority checkpoint, `ablate.py` compares reference, `mlxboost_b1`
flag-off, and the single `fast_decode` flag. Exact token-ID lists, decoded text
(including whitespace), terminal token IDs, finish reason and generated counts
match. Captured logits and KV are bit-identical, rather than merely within a
tolerance. There is no new sampler or detokenizer in this change.

The full original-protocol current-source HTTP flag-off versus flag-on comparison
also matches exact visible bytes, completion counts and finish reasons on all
6/6 prompts for each of Qwen2.5, Qwen3 and SmolLM2. The SSE benchmark does not
expose authoritative raw generated IDs; those are compared in the native Engine
trace. SSE chunk counts and retokenized text are never used as generated IDs.

Upstream text differences remain unresolved as acceptance gates. The pinned
MLX-LM BPE detokenizer explicitly removes an initial space; MLX-One's raw tokenizer
decode retains it. The historical 0/6 exact / 6/6 trimmed Qwen agreement is a
diagnostic observation, not a production trimming rule. This patch preserves
native whitespace exactly. SmolLM2 additionally has a historical EOS divergence
on prompt index 2, retained in both filtered and unfiltered comparison records.

Numerical comparison on equal forward partitions is distinct from upstream
generation equivalence. MLX-LM's generation prefills all but the last prompt token
before its first sampling step; native generation uses the full prompt. The
existing cached/fresh failures demonstrate why those partitions must be examined
before declaring exact upstream generation qualification. No alpha model status
or precision gate was changed.

The matched-MLX 0.32.2 native source probe replays full-prefill argmax,
127+1-prefill argmax and pinned asynchronous `generate_step` on prompt indices
1 and 2. All three variants produce identical IDs for both cases in each
priority model; SmolLM2 produces 128 tokens on index 2 in this diagnostic.
Thus the shorter EOS observed in the MLX 0.32.4 whole-stack HTTP comparison is
not reproduced by this matched-backend native probe. Backend and serving
differences remain confounders; no unique EOS cause is claimed. Raw source IDs,
termination and timings are in `source-requalification-20261101/reference_probe-*/`.

Expanded process-paired agreement and current API cancellation/usage checks are
reported in `benchmark-comparison.md` and `api-regression.md`, with their raw IDs.
