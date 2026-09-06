# Contributing to mlx-one

Thank you for helping build reproducible model workflows targeting Apple Silicon.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

Use Python 3.10 or newer. Public functions and classes require type hints and
docstrings. Keep core modules importable without initializing Metal, and put
modality-specific dependencies behind extras.

## Changes and tests

- Open an issue before starting a large feature or changing a public interface.
- Add unit tests for behavior and failure modes. Unit tests must not download
  models or datasets.
- Mark real-model checks as integration tests and document the required hardware,
  model revision, and expected memory use.
- Pin model, dataset, prompt, and evaluator revisions in reference results.
- Run `ruff check .` and `pytest -q` before opening a pull request.

Pull requests should describe the user-visible outcome, tests performed, and any
result-schema or compatibility implications. Do not claim a roadmap milestone as
complete until all of its exit criteria are demonstrated.
