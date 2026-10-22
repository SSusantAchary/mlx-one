# Single-owner source map

20 October 2026 owner-directed Qwen3 adoption overrides the earlier A4 math-preservation
restriction only for the parallel `models/language/qwen3/upstream*.py` candidate and its
Qwen3-owned loader/cache bridge. Original `qwen3/model.py` and shared mathematics remain
unchanged until explicit promotion gates pass. Candidate integration hooks in `text/loading.py`,
`engine/modes.py`, `engine/public.py` and `server/generation_engine.py` belong to this adoption
task; they preserve existing defaults and reject unqualified candidate execution modes.
The existing A10 TaskProfile/worker gets one explicit Qwen3 candidate selector so frozen
quality fixtures can execute through the same qualification framework without registry mutation.
New regressions and optional native tests use `tests/alpha_0_1`; qualification preparation
reuses the existing alpha/task infrastructure. MIT source notice/package-data accounting
belongs to A9; inclusion remains pending and no distribution build is authorized.

| Paths | Owner | Permitted action |
|---|---|---|
| docs/alpha_0_1 baseline/ownership/inventory/folder/progress | A0 | Add audit only |
| engine/modes.py, ops/attention.py, experimental adapters and isolation tests | A1 | Verified minimal fix only; no moves planned |
| schemas.py, schema_migrations.py, core/precision.py and precision tests | A2 | Add exact format descriptions; retain serialized 1.0 compatibility |
| core/registry.py, models/registry.py, engine/adapter_registry.py/runtime_specs.py | A3 | Extend existing metadata only if needed |
| text/**, models/language/**, text tests and examples | A4 | Loader correctness; preserve native math |
| retrieval/**, models/embeddings/**, retrieval tests and examples | A5 | Pooling and task validation |
| audio/**, models/audio/**, audio tests and examples | A6 | Safe preprocessing and native ASR |
| vision/**, models/vision_language/**, vision tests and examples | A7 | Input validation and processor correctness |
| __init__.py, cli.py, server/** and shared facade tests | A8 | Preserve public imports/routes; harden defects |
| pyproject.toml, packaging manifests, README support only, docs/examples/CI | A9 | Distribution audit; no unapproved source packaging |
| scripts/qualification/alpha/**, qualification runner narrow extensions, support/acceptance/report docs | A10 | Reuse native_tasks and p1 qualification |
| historical optimization docs/results, unrelated project, generated dependencies | None | Preserve unchanged; never release |

Every future new test/example/script uses its stage's existing directory or `tests/alpha_0_1`, `examples/alpha_0_1`, `scripts/qualification/alpha`. These additive paths are assigned here before creation. No new registry, HTTP app, scheduler or embedding runtime is authorized. All task implementation/validation docs belong to their stage; progress/report entries are updated at each handoff. Cross-owner defects must be queued in progress.
