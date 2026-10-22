# A5 embedding implementation and validation

Fixed observable metadata errors: sentence encoders previously ran mean pooling regardless of configured pooling type and derived normalization from the mean-pooling flag. They now reject unsupported pooling/dimension/module pipelines before native allocation and honor optional Normalize modules separately. Legacy assets without modules.json keep the existing normalize_embeddings config/default. BERT/MPNet mathematics, masking and default pooling implementations remain unchanged. Qwen last-token/Matryoshka, rerank pair scoring and ColBERT late-interaction remain distinct.

`embed` now accepts max_length=None (new default), choosing min(8192, native capacity) so ordinary MiniLM/MPNet calls do not fail at the old 8192 default. Explicit excessive lengths still reject. Retrieval safetensors reuse the CPU finite validator/direct native reader, preserving BF16 rather than converting it through NumPy.

**53 passed**, 0.85s; command: `PYTHONPATH=src:tests:. .venv/bin/python -m pytest --no-cov -q tests/alpha_0_1/test_embedding_contract.py tests/test_native_sentence_transformers_architectures.py tests/test_native_qwen3_retrieval.py tests/test_task_retrieval_runtime.py`. Coverage: supported configs/weights, pooling metadata negatives, normalization on/off, Unicode/default length, vector dimensions, task mismatch, independent/batched ordering, masking and task lifecycle. Focused Ruff PASS; example --help PASS. Logs/JUnit: ignored a5 files.

Candidate only: real vector quality/golden cosine thresholds, checkpoint hashes, Python/CLI/HTTP same-checkpoint M4 smoke and precision remain HARDWARE_PENDING/NOT_QUALIFIED. No vector outputs or performance fabricated. Existing native task runner supports independent embedding profiles; A10 supplies bounded offline invocation. Empty strings retain the existing explicit input error policy.

A10 owner-mediated A5 correction: sentence/ColBERT quantization metadata is now checked before tokenizer, weight reads or model allocation; added two no-assets negative regressions. Full final suite validates the correction without running a checkpoint.
