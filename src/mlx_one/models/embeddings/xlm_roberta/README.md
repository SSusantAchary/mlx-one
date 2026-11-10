# Native XLM-R retrieval

The encoder reuses compatible BERT attention/FFN primitives with XLM-R's
non-padding position arithmetic and tokenizer semantics. Reranking uses the
trained dense/tanh/output classification head and returns raw scalar logits.

The safely converted BGE-M3 profile adds trained dense, positive sparse and
ColBERT token heads with explicit hybrid scoring. BGE reranker v2 M3 uses the
paired classification profile. Those are two checkpoints under one architecture
ID; their tasks and trained heads are not interchangeable.

See the [model workflow guide](../../../../../docs/models/batch-1-sub-6b/README.md)
for safe conversion, representation semantics, limits and source results.
