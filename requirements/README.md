# Dependency policy

`pyproject.toml` declares install requirements and optional extras.
`alpha-tested-constraints.txt` preserves exact versions from the recorded Apple M4,
macOS, Python 3.12.11 candidate environment. A constraints file limits resolution;
it does not install packages by itself or certify other interpreters.

## Tested dependency minimums

The following lower bounds use the versions in that recorded candidate environment.
They are conservative tested floors, not claims about the earliest historical
version that could work. Each package's installed `Requires-Python` metadata permits
Python 3.10, but full runtime compatibility still needs testing on each interpreter.

| Dependency | Declared requirement | Reason |
| --- | --- | --- |
| tqdm | `>=4.70.0` | Recorded progress-reporting dependency |
| safetensors | `>=0.8.0` | Recorded tensor loading, validation and serialization dependency |
| PyYAML | `>=6.0.3` | Recorded safe YAML configuration dependency |
| regex | `>=2026.9.3` | Recorded text/tokenizer dependency |
| Pydantic | `>=2.13.5,<3` in `server` and `dev` | Recorded v2 dependency; HTTP schemas directly use v2 APIs |

The four core packages have no additional upper caps because no specific
incompatibility is recorded here. Add a cap when a reproduced incompatibility
justifies it. Pydantic is restricted to major version 2 because the server uses
its v2 schema and validator APIs.

## Installing and reproducing environments

For the recorded candidate, first create and activate a Python 3.12.11 virtual
environment on Apple Silicon macOS, then run from the repository root:

```bash
python -m pip install -c requirements/alpha-tested-constraints.txt -e ".[dev]"
python -m pip check
```

For Python 3.10, 3.11 or 3.13, use a separate virtual environment and let pip resolve
that interpreter's compatible dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest --no-cov
```

Do not reuse the Python 3.12 snapshot as a universal lockfile. If reproducible pins
are needed for another interpreter/platform, capture and validate a separate
constraints file. Keep the existing candidate snapshot intact until a new candidate
is deliberately tested and recorded. Dependency resolution and `pip check` alone
do not establish native Metal execution or model qualification.

## Optional Qwen alignment dependencies

| Extra | Exact tokenizer requirement | Additional asset |
| --- | --- | --- |
| `qwen-align-ja` | `nagisa==0.2.11` | None added by mlx-one |
| `qwen-align-ko` | `soynlp==0.0.493` | Checksummed `korean_dict_jieba.dict` in the local checkpoint root |

These tokenizers are deliberately separate from core, `audio`, `server` and `dev`.
They were not installed in the recorded candidate snapshot and have not been
qualified across all advertised Python versions. See the
[Qwen installation and asset instructions](../src/mlx_one/models/audio/qwen3_asr/README.md#installation-and-assets)
for installation commands, the dictionary hash and runtime validation behavior.
