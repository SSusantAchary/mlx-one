# Documentation generation and validation

Run these commands from the repository root in the development environment.
They read existing records without downloading weights or running inference.
JSON inventories, original measurements, checkpoint locks and qualification
decisions remain the authoritative evidence.

## Generated-region ownership

| Destination | Marker pair | Owner |
| --- | --- | --- |
| `README.md` | `model-performance-summary:start/end` | `benchmarks/model_performance/report.py`: compact primary-precision table from one historical cohort |
| `docs/benchmarks/text-performance.md` | `model-performance:start/end` | Same text reporter: all historical and standard cohorts, separately labeled |
| `README.md` and `docs/benchmarks/vlm/performance.md` | `MLX-ONE-VLM-BENCHMARK-START/END` | `benchmarks/vlm/readme_generator.py` |
| `README.md` and `docs/benchmarks/audio/performance.md` | `MLX-ONE-AUDIO-BENCHMARK-START/END` | `benchmarks/audio/readme_generator.py` |
| `README.md` and `docs/benchmarks/ocr/performance.md` | `MLX-ONE-OCR-BENCHMARK-START/END` | `benchmarks/ocr/report.py`, with existing corpus renderer |
| `docs/experimental/native-tasks.md` | `mlx-one:verified-task-results:start/end` | Editorial publication boundary; no active generator found. Claims checked against existing task reports |
| `docs/experimental/mlxboost.md` | `mlx-one:mlxboost-summary:start/end` | Editorial publication boundary; no active generator found. Existing ablation/publication evidence owns the numbers |
| `docs/benchmarks/inventory.md`, `coverage.md`, `coverage.json` | Whole file | Text reporter; authoritative inventory JSON and support manifest are inputs |
| Modality coverage files | Whole file | Existing modality reporters and selected publication records |
| `docs/benchmarks/alpha-candidates/RUN/performance.md` | Whole file, when an alpha candidate is packaged | `scripts/qualification/alpha_package.py`; keeps each candidate run separate instead of appending tables to README |

The shared presentation helper rebases repository-root renderer links to each
report's directory. It changes neither source data nor metric selection. Marker
replacement requires exactly one ordered pair and never silently inserts a new
experimental status section into README. Measured modality tables and charts belong in README. The VLM generator retains its legacy `--readme`
option for custom destinations; its default refreshes both README and the companion VLM performance report.

## Regeneration

```bash
python benchmarks/model_performance/report.py --write
python -m benchmarks.vlm.readme_generator
python -m benchmarks.audio.readme_generator --write
python -m benchmarks.ocr.report --write
```

Repeat these commands and compare file hashes: the second run must change
nothing. The homepage retains text, VLM, audio and OCR performance plus separately
labeled HTTP/task measurements and explicit unmeasured modality gaps. Experimental
qualification tables, blockers and optimization investigations stay in dedicated docs. Generated
inventory, coverage and original evidence hashes must remain stable when the
evidence selection has not changed. Chart generators keep their existing paths
and evidence; all image assets and historical charts are retained.

## Checks

```bash
python benchmarks/model_performance/report.py --check
python -m benchmarks.vlm.readme_generator --check
python -m benchmarks.audio.readme_generator --check
python -m benchmarks.ocr.report --check
python scripts/release/check_documentation.py
python -m benchmarks.audio_documentation_check
python -m unittest discover -s tests/unit -p test_model_performance_reporting.py
pytest -q tests/test_vlm_benchmark.py tests/test_audio_benchmark.py tests/ocr/test_ocr_benchmark.py
```

The standard-library link checker validates local targets, heading/HTML anchors,
images, generated-marker ownership, tables, fences and Python example syntax. It
also requires the modality performance markers and rejects the removed checkpoint
status table and experimental qualification markers in README. External
URLs are counted but not requested. Audio documentation checks find model-guide
coverage through the canonical model index, rather than requiring every family
guide to appear on the landing page. Model-performance CI runs the portable
link/ownership check and existing evidence/unit checks on documentation changes.

## Historical publication audit

The [2026-11-04 publication audit](publication/2026-11-04/audit.md), evidence map
and preview are retained historical artifacts. Its full audit verifies frozen
implementation, protected-file and Git-index hashes from that campaign; it must
not be used to approve a different working-tree baseline. Its `--self-test` and
`--markdown-only` modes follow the relocated reports. Preview output goes to the
ignored local audit run directory so the historical preview is preserved.

Current refactor validation is recorded in the
[restructuring results](readme-restructuring-results.md), relative to the separate
2026-11-07 baseline. No benchmark, qualification or publication campaign is rerun.
