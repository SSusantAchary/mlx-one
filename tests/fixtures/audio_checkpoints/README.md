# Metadata-only audio checkpoints

These JSONs contain pinned public configuration and safetensors **header** shapes
and dtypes. They contain no trained weight payloads or speech samples. Remote
inspection used bounded HTTP Range requests; no full weights were downloaded.
Each JSON records the exact repository and commit SHA used on 2026-10-24.

Qwen ASR/alignment metadata is from the listed mlx-community conversions of Qwen's
Apache-2.0 models. LFM metadata refers to LiquidAI's separately licensed model;
source adaptation attribution is in the model's `UPSTREAM.md`/`UPSTREAM_LICENSE`.
Parakeet metadata is from the listed MLX conversion of NVIDIA's CC-BY-4.0 model;
retain the native Parakeet `NOTICE` and model-card attribution. Vocabulary pieces
and structural metadata are test contracts, not a redistribution of model weights.

`test_audio_pipeline_native.py` checks actual module names, shapes, packed widths,
codec vocabulary distinctions and convolution layouts against these records.
Engineered tiny checkpoint weights are generated only in pytest temporary folders.
Their deterministic transcripts/audio test control flow and resource ownership;
they are never used as performance, intelligibility or real-checkpoint evidence.
