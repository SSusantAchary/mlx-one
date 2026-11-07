# Development and validation

## Validation

The current repository gates include:

- Backend-free tests covering schemas, configs, registries, weight contracts,
  tokenization, processing, evaluation, loading policy, and CLI behavior.
- Opt-in Apple Silicon/Metal tests covering native model execution, shapes, caches,
  masks, multimodal feature insertion, embeddings, reranking, MoE routing, and ASR
  components.
- Pinned real-checkpoint smoke gates for `openai/whisper-tiny` and
  `openai/whisper-large-v3-turbo`.
- Whisper log-Mel comparison within `1e-5` against the pinned reference path.
- SAM3 network-free prompt/export/lifecycle checks, strict pinned checkpoint
  loading, and separately recorded 32-frame offline/streaming CPU-reference
  fixture comparisons. Natural multi-object tracking qualification is pending.
- Ruff, source/wheel builds, and package metadata checks.

These counts describe the current development tree and will change as coverage
expands.

Run backend-free checks:

```bash
python scripts/release/check_lint.py
pytest -q
python -m build --no-isolation
python -m twine check dist/*
```

Run native MLX execution tests on an Apple Silicon host:

```bash
MLX_ONE_RUN_MLX_TESTS=1 pytest -q
```

Run pinned Whisper integration gates:

```bash
MLX_ONE_RUN_WHISPER_INTEGRATION=1 pytest -q \
  tests/test_native_whisper_integration.py
```

Ordinary tests are network-free. Integration tests may download only the pinned
assets needed by their explicit gate.

## Roadmap

The implementation proceeds by evidence-backed vertical slices:

1. Complete checkpoint loading, generation, and parity qualification for native
   language families.
2. Expand the native retrieval APIs beyond the initial Qwen3 embedding and
   reranking vertical slice and qualify exact checkpoint revisions.
3. Add complete image processing, generation, OCR/VQA evaluation, and training
   paths for native VLMs.
4. Qualify Whisper revisions and expand native ASR, alignment, audio-language,
   and TTS coverage.
5. Add native quantization, advanced training, export, and community evidence
   workflows.

See the [development roadmap](roadmap.md) for the broader plan.
