# Native NomicBERT embeddings

Checkpoint-compatible fused QKV attention, split-half rotary positions, SwiGLU
and post normalization execute in MLX. Publisher Python code runs only in the
isolated development oracle; native loading never enables remote code.

The admitted Nomic Embed v1.5 profile uses role prefixes and a normalized 768-D
default. Explicit Matryoshka dimensions apply full-vector layer normalization,
then truncation and normalization, including explicit dimension 768. Default
and explicit-width requests have distinct actor batching identities.

See the [model workflow guide](../../../../../docs/models/batch-1-sub-6b/README.md)
for tested lengths, local prefix sidecars, APIs and all five dimensionality checks.
