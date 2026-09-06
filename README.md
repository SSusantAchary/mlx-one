# mlx-one

Train anywhere. Prove it on Apple Silicon.

`mlx-one` is building an open, model-size-agnostic lifecycle for training through
pluggable backends, converting models to MLX, evaluating text, vision-language,
ASR, and TTS models, comparing quality and device performance, and packaging the
evidence needed to ship a model.

> Status: `v0.1.0a1` foundation. The diagnostic and model-loading utilities are
> available; evaluation, comparison, merging, and release bundles are milestone
> work and are not advertised as complete.

## Development installation

```bash
git clone https://github.com/SSusantAchary/mlx-one.git
cd mlx-one
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional model backends are installed independently:

```bash
python -m pip install -e ".[vlm]"         # mlx-vlm
python -m pip install -e ".[audio]"       # mlx-audio
python -m pip install -e ".[embeddings]"  # mlx-embeddings
python -m pip install -e ".[multimodal]"  # all modality backends
python -m pip install -e ".[tracking]"    # optional local MLflow integration
```

## Check your Mac

Run diagnostics without importing MLX into the CLI process:

```bash
mlx-one doctor
mlx-one doctor --json-output
```

The report covers the host architecture, chip, unified memory, Python version,
Metal availability, and installed MLX modality backends. MLX model execution is
supported on Apple Silicon macOS. Model size is not an eligibility restriction;
compatibility evidence is attached to the exact model, configuration, workload,
and hardware used for each verified run.

## Current Python utilities

```python
from mlx_one.utils.model_loader import (
    load_audio_model,
    load_embedding_model,
    load_model,
    load_vlm_model,
)

model, tokenizer = load_model("mlx-community/Qwen3-0.6B-4bit")
vlm, processor = load_vlm_model("mlx-community/Qwen2-VL-2B-Instruct-4bit")
asr = load_audio_model("mlx-community/parakeet-tdt-0.6b-v3", task="stt")
embedding_model, tokenizer = load_embedding_model(
    "mlx-community/all-MiniLM-L6-v2-4bit"
)
```

Optional loaders raise an actionable error identifying the required package extra
when their backend is missing.

## Roadmap

| Milestone | Outcome | Status |
| --- | --- | --- |
| M0 / `v0.1.0a1` | Reliable foundation and diagnostics | 🚧 in progress |
| M1 / `v0.2.0` | Unified text quality and performance | 📋 planned |
| M2 / `v0.3.0` | Vision-language evaluation | 📋 planned |
| M3 / `v0.4.0` | ASR and TTS evaluation | 📋 planned |
| M4 / `v0.5.0` | MLX-native text-model merging | 📋 planned |
| M5 / `v1.0.0` | Quality gates and release bundles | 📋 planned |

Milestones advance only after their implementation, validation, and documentation
are complete. Unfinished functionality remains experimental and is not presented
as shipped.

## Development checks

```bash
ruff check .
pytest -q
python -m build
python -m twine check dist/*
```

Model-loading and reference benchmark runs require a real Apple Silicon terminal
with Metal access and are kept separate from backend-free unit tests.

## Contributing and security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a
pull request. Report security issues using [SECURITY.md](SECURITY.md), not a public
issue.

Released under the [MIT License](LICENSE).
