# Native T5

Encoder-decoder text-to-text generation with shared token embeddings, T5 RMS
normalization, unscaled query/key attention, relative position buckets and
encoder/cross/causal decoder masks. Greedy decoding reuses the encoder and
recomputes bounded decoder prefixes; no incremental KV-cache claim is made.

The pinned T5-small profile passes teacher-forced logits and deterministic
generation against the independent source. Use `mlx_one.seq2seq.generate` or
`mlx-one seq2seq`; this model does not use causal-chat APIs.

See the [model workflow guide](../../../../../docs/models/batch-1-sub-6b/README.md#t5-encoder-decoder-text-generation)
for input bounds, Engine/HTTP examples and exact checkpoint evidence.
