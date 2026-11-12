# Batch 1 implementation plan

Started 9 November 2026 from `d922903`, including the local CI compatibility fixes.
Original registry: 40 types. Baseline: 2,254 passed, 118 skipped, 9 subtests passed.
Host: Apple M4, 32 GiB, macOS 15.8.2, Python 3.12.11, MLX/Metal 0.32.2.
The original guidelines worktree remains untouched. No publication is authorized.

Follow [development guidelines](../../../mlx_one_development_guidelines_v1.0.md).
The requested batch is ten **checkpoint targets**, not ten extra architecture IDs.
Existing 40-model approval remains separate from new checkpoint admission.

## Gap analysis and execution order

| Target | Existing reuse | Missing work | Verification |
| --- | --- | --- | --- |
| B01 MiniLM cross-encoder | Native BERT encoder, tokenizer, retrieval APIs | Pooler + scalar head, paired segment IDs, task-aware service/adapter | Pinned pair logits, ranking, batching, Python/CLI/HTTP |
| B02 BGE-small | Native BERT encoder | CLS pooling alongside mean; honor Normalize module | Pinned vectors, masks, norms, retrieval ranking |
| B04 multilingual MiniLM | Native BERT encoder | Preserve raw mean output and 128-token sentence bound | Multilingual vectors, padding, source parity |
| B10 multilingual E5 | Native BERT encoder | Explicit query/passage prefixes, mean + normalize | Multilingual retrieval, prefix and batching parity |
| B07 BGE reranker | BERT attention/FFN | XLM-R position semantics and classification head | Pinned multilingual pair logits and ranks |
| B03 BGE-M3 | XLM-R backbone from B07 | Safe offline conversion; dense/sparse/ColBERT heads and scoring | Separate head parity, token filtering, multilingual ranking |
| B09 Nomic | Shared masking/normalization | Native fused-QKV/RoPE/SwiGLU encoder, prefix and Matryoshka pipeline | Publisher tensor and truncation/normalization parity |
| B06 ViT | Existing vision/attention primitives | ViT encoder/classifier and exact image processing | Public images, logits/top-k parity |
| B05 T5 | Attention/norm primitives | Encoder-decoder, relative biases, teacher forcing and generation | Pinned logits and greedy sequence parity |
| B08 Kokoro | Audio I/O and task contracts | Safe conversion, ALBERT/acoustic/vocoder math, phonemes/voice | Acoustic-stage parity, waveform/listening, 24 kHz and RTF |

## Evidence and resources

- [Baseline and pinned inventory](../../qualification/batch-1-sub-6b/) record
  exact revisions, configs, model-card licenses, parameter metadata and baseline.
- Run the reference only in an isolated environment. Runtime imports remain
  native MLX, NumPy and declared tokenization/processing dependencies.
- Use one native GPU workload at a time. Keep downloads under a dedicated
  ownership-marked temporary root; secure result/hash records before deleting
  each completed model. Never delete shared caches or protected user models.
- Preserve exact source/raw precision for first admission. Quantization and
  long-context promotion require their own actual runs.
- Record real checkpoint, source parity, task quality, interfaces, benchmark and
  distribution gates separately. A blocked optional scope must remain explicit.
