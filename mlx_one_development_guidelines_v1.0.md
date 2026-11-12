# MLX-One development guidelines v1.0

Version: **1.0** · Established: **9 November 2026** · Owner: **project maintainer**.

These guidelines define how to grow MLX-One and accept future changes. They
consolidate the native implementation approach, functional model admission,
resource cleanup, packaging, and documentation practices used in the recent
release work. They do not change runtime defaults or issue a new release approval.

**Working standard:** deliver a complete native workflow, test the behavior being
claimed, retain reproducible evidence, and keep public documentation consistent.

## 1. Policy and current baseline

- Follow the current [model support policy](docs/qualification/model-release-20261108/SUPPORT_POLICY.json)
  and [model release guide](mlx-one_model_release_guide.md).
- The approved baseline is **40 model types across seven modalities**, with
  checkpoint-specific primary workflows listed in [support status](status_models_testing.md).
  This is a dated baseline, not a fixed limit or a claim about every checkpoint.
- **SUPPORTED** means implementation tests, real checkpoint execution, relevant
  public interfaces, regressions, and distribution checks passed for the stated
  scope. Do not impose an automatic experimental label or an additional general
  reliability certification on those supported workflows.
- Corpus accuracy, universal numerical tolerances, all-device comparisons, and
  optional task extensions are not universal support gates. They become
  acceptance criteria when the change explicitly claims that quality, numerical
  behavior, hardware coverage, or extension.
- Loading failures, invalid task outputs, incorrect state, broken promised
  interfaces, and regressions remain blockers. Optional diagnostics cannot excuse
  those defects. Historical failed measurements must remain failed.
- Use the current support catalog and release decision for current claims.
  Pre-campaign plans, old restricted-alpha manifests, and dated audit reports
  retain their original scope; they do not override later reviewed decisions.

## 2. Choose work that grows usable coverage

| Priority | Work | Acceptance outcome |
| --- | --- | --- |
| P0 | Regressions, incorrect loading/output/state, broken installation or supported interfaces | Restore the affected supported workflow and add a regression check |
| P1 | New model types, checkpoint profiles, or missing primary modality workflows | Complete native implementation, real checkpoint task, relevant interfaces, evidence, and documentation |
| P1 | Shared processors, loaders, registries, and developer usability | Demonstrate the affected workflows and preserve existing contracts |
| P2 | Performance, batching, memory, and optional acceleration | Reproducible benefit with the required correctness and lifecycle checks |
| P2 | Coverage summaries, model explorer, benchmarks, and development milestones | Derive displays from maintained manifests and recorded results |
| P3 | New training methods, speculative execution, or broader research claims | Explicit scope, feature status, and method-specific acceptance before promotion |

Prefer a complete, useful vertical slice over registering many unfinished
architectures. Reuse existing evidence and shared components when their identities
and tested behavior still apply. Do not download new models merely to populate a
dashboard or demonstrate that development activity occurred.

## 3. Define the change before implementation

For each substantial change, record:

1. The user problem and observable result, with a concrete input/output example.
2. The model type, task, checkpoint/precision, public interfaces, and input bounds
   involved; identify any excluded extensions.
3. The affected shared components and existing workflows that could regress.
4. Acceptance checks, fixtures, expected behavior, and any claim-specific
   thresholds. Set thresholds before measuring the claim.
5. Hardware, memory, storage, optional dependencies, and required model access.
6. Documentation, evidence, compatibility, and release implications.

Discuss large features and public interface changes through an issue or design
record, following [CONTRIBUTING.md](CONTRIBUTING.md). Routine fixes can use a
focused pull request. A plan or source registration alone does not complete a
milestone.

## 4. Preserve the native architecture

- Native inference remains inside `mlx_one` on Apple MLX. Keep model equations,
  processors, checkpoint loading, tokenization, and execution under their existing
  package owners.
- Use configuration and registry metadata to select architectures and tasks.
  Avoid model-name shortcuts and duplicate dispatch authorities.
- Reuse shared attention, normalization, RoPE, masking, MoE, cache, and processing
  primitives where their semantics match. Keep family-specific behavior in the
  relevant model package.
- Keep configuration, weight conversion, preprocessing, model forward,
  generation, serving, and training responsibilities separate.
- Reuse the model implementation between inference and training where supported.
  A new training path must not introduce a divergent inference model silently.
- Keep lightweight imports, registry listing, configuration, and artifact
  inspection usable without initializing Metal or downloading assets.
- Upstream MLX libraries, Transformers, and Torch may be isolated conversion or
  validation references. Do not add them to the native production path just to
  obtain a passing result. Keep the existing explicit legacy extra separate.
- Preserve explicit execution selection and recorded fallback reasons. A request
  for an unsupported native capability must not silently run a different model,
  precision, task, or foreign runtime.

See the [architecture](docs/architecture/index.md) and
[package ownership](docs/REPOSITORY_SUMMARY.md) for existing boundaries. Dated
implementation notes in those documents do not supersede current support status.

## 5. Configuration, weights, and conversion

- Validate configuration, tensor names, dimensions, dtypes, masks, and architecture
  options. Reject unsupported settings and unexplained missing or extra tensors.
- Make weight mapping explicit and deterministic. Document legitimate aliases,
  tied weights, ignored source tensors, and conversion steps.
- Sanitizing an already sanitized bundle must not corrupt it. Exercise relevant
  save/load and conversion round trips.
- Verify tokenizer vocabulary, BOS/EOS IDs, chat template, processor settings,
  and the checkpoint's required assets together.
- Record storage precision separately from execution and cache precision. Affine
  Q4/Q8, MXFP4, and TQ3 require their own loading and numerical contracts; similar
  filenames do not make formats interchangeable.
- Use safe tensor loading for native checkpoints. Isolate any necessary legacy
  conversion, identify trusted source bytes, and retain provenance. Do not enable
  arbitrary remote code as a convenience fallback.
- Hash actual model/config/processor bytes with bounded-memory reads. Parameter
  count and a model name are insufficient checkpoint identity.

## 6. Model admission gates

Apply these gates to the advertised model profile. A support claim identifies:

`model type + checkpoint + immutable revision + asset/conversion identity + precision + task + interface + environment + input bounds`

| Gate | Required work | Acceptance criterion |
| --- | --- | --- |
| G0 — Scope and identity | Record checkpoint revision, assets, conversion, precision, fixtures, source identity, environment, and intended operations | Every claimed result can be traced to the actual inputs and implementation |
| G1 — Implementation contracts | Test tiny native forward behavior, shapes/dtypes, masks, state/cache, loading, and relevant conversion/failure paths | Applicable assertions pass; random weights only establish implementation behavior |
| G2 — Real checkpoint task | Load at least one exact checkpoint for the advertised profile and save actual task input/output | Native execution performs the named task with valid output; loading alone is insufficient |
| G3 — Relevant interfaces | Exercise each advertised Python, CLI, HTTP, streaming, or session path | Promised operations work and invalid inputs fail clearly; nonexistent interfaces need no artificial test |
| G4 — Regression and distribution | Run affected regressions, native checks where needed, and appropriate installed-artifact checks | The change preserves existing supported behavior and the installed package contains the tested implementation/assets |
| G5 — Evidence and review | Save results/hashes, limits, cleanup receipt, usage, documentation, and maintainer decision | Public claims match the reviewed evidence; task-owned downloaded weights are removed after results are secured |

Existing evidence can be reused if the relevant implementation and asset identity
still match. Rerun affected gates after a change; shared runtime changes require
coverage of the affected execution paths. A metadata or prose edit does not
require downloading every model again.

## 7. Task acceptance by modality

Choose fixtures for the actual model task. A generated tensor with the right
shape, source file, or mock response does not replace a real checkpoint workflow.

| Modality/task | Minimum functional acceptance | Additional checks only when claimed |
| --- | --- | --- |
| Text generation | Correct tokenizer/template and BOS/EOS handling; real completion; prefill/decode, continuation/reset, and finite outputs | Long context, tool calls, batching, speculative decoding, extra precisions |
| Vision-language | Known image and factual prompt; correct resize/grid/position/token handling; saved answer tied to image identity | Multi-image, video, document QA, resolution or context expansion |
| Audio: ASR | Correct waveform/frontend/sample rate; saved transcript from real speech; silence/short-input handling and session state | Languages, translation, timestamps, streaming, corpus WER/CER claims |
| Audio: forced alignment | Known transcript with independently timed speech; ordered boundaries inside audio; matched/unmatched-word accounting | Specific boundary-error targets, languages, or broader corpus accuracy |
| Audio: TTS / speech-to-speech | Finite, nonempty waveform at the declared sample rate; bounded amplitude/duration and intelligible task output; real spoken input for STS | Streaming chunk order/finalization, languages, TTFA/RTF, listening-study claims |
| Audio: VAD | Speech and silence fixtures; frame/context/recurrent state, bounded timestamps, reset/finalization and independent sessions | Annotated boundary quality, languages or noise robustness |
| OCR and documents | Readable real line or page suited to the model; correct processor/prompt/decoding; saved recognizable text | Layout, regions, tables, formulas, full-page coverage or corpus CER/WER |
| Embeddings | Finite vectors with correct dimensions, pooling/masking/normalization; saved relevant/dissimilar examples and batch consistency | MTEB, retrieval scores, languages, dimensionality or length extensions |
| Reranking / ColBERT | Correct pair template or token-vector/MaxSim contract; padding exclusion, score direction, and meaningful fixture ranking | Corpus retrieval metrics, batching limits, extra query/document profiles |
| Segmentation | Real masks with correct image transforms and prompt handling; valid output geometry | Automatic masks, additional prompt modes, corpus IoU or other quality targets |
| Video segmentation / tracking | Real frames, temporal IDs, coordinate consistency, and session lifecycle; offline/streaming behavior for the advertised path | Natural multi-object coverage, longer videos, additional reference parity |
| Classification / NER / CLIP | Correct label mapping and input bounds; appropriate class output, valid entity spans/aggregation, or image/text candidate matching | Dataset accuracy/F1, multilingual behavior, calibrated confidence |

Subtasks are not interchangeable: transcription does not qualify alignment; line
OCR does not qualify page layout; sentence embeddings do not qualify pair
reranking; image segmentation does not qualify temporal tracking. Record a
separate result for each promised task or materially different profile.

## 8. Hugging Face fixtures and reference runs

- Prefer a small, suitable public Hugging Face dataset sample when available.
  Record dataset ID, immutable revision or file hash, configuration, split,
  selected row IDs, preprocessing, and permitted use.
- Use representative task inputs with known labels/annotations where needed.
  Preserve sample hashes and expected behavior. A generated unit fixture is
  useful for contracts but must be labeled synthetic.
- Keep normalization, evaluator version, prompts, generation parameters, and
  random seeds in the result. Avoid tuning a quality claim against its test set.
- Use model-family references already identified in the
  [release guide](mlx-one_model_release_guide.md). Pin the reference implementation
  and run it separately from the native environment.
- Compare meaningful task behavior, tensors, or state as appropriate. Cross-runtime
  generated prose need not be identical unless an exact-output claim is made.
- Keep CER/WER/F1/PPL, retrieval, mask, and reference discrepancies visible as
  measured. A policy change must not rewrite those scores or failure statuses.
- An HTTP 403, gated tokenizer, missing asset, or resource limit is a recorded
  blocker for that profile. Continue independent models; use an authorized,
  identity-verified local asset when available. Do not substitute a different
  tokenizer or mark an unexecuted required check as passed.

## 9. Public APIs and request lifecycle

- Public functions/classes require type hints, docstrings, clear input bounds,
  and actionable errors. Keep Python, CLI, and HTTP semantics consistent for
  operations exposed through more than one interface.
- Reuse the existing Engine/service/adapter boundaries. Preserve result schemas,
  token/sample/frame usage, finish reasons, and requested/resolved configuration.
- Test affected stop/EOS behavior, streaming finalization, cancellation,
  exceptions, context-manager cleanup, and independent request/session state.
- Prevent state leakage across users, requests, and model leases. Release cache,
  memory reservations, handles, and temporary resources on completion and failure.
- Maintain backward compatibility. Explain intentional breaking changes, migration,
  and versioning implications before accepting them.
- Training, adapters, checkpoint export, and speculative features need their own
  affected correctness/save-load/restart checks. Their presence in source does
  not expand an inference support claim.

## 10. Tests, Python compatibility, and dependencies

- Ordinary unit tests are network-free and download no weights or datasets.
  Keep backend-free contracts separate from explicit native/integration checks.
- Add meaningful regression tests for behavior changes and reproduced defects.
  Do not add tests that merely copy the implementation or perform unrelated
  research. Documentation-only changes need documentation checks.
- Use a fresh virtual environment and `python -m pip` / `python -m pytest` so
  installation and tests use the same interpreter. Check `python -m pip check`.
- The currently advertised Python classifiers are **3.10–3.13**. Runtime,
  dependency, and packaging changes must be checked against those advertised
  versions as applicable; docs edits do not require repeating the full matrix.
- Respect the Python 3.10 floor: use compatible datetime and hashing APIs, and
  an explicitly declared TOML fallback where needed. Compilation alone does not
  demonstrate import, collection, or execution compatibility.
- Declare test and build dependencies explicitly. Build isolation does not
  install build-backend dependencies into a pytest environment.
- Keep modality and server dependencies behind their appropriate extras.
  Declare dependency bounds from reproduced compatibility evidence; do not
  treat one Python 3.12 constraints snapshot as a universal lockfile.
- Preserve the tested MLX/Metal pairing. A backend/version upgrade requires
  affected native execution and package/import checks, not only resolver success.
- Run the repository lint gate. Do not expand inherited lint debt, suppress new
  findings broadly, or change tests/thresholds merely to obtain a green result.
- Record pass, fail, skip, blocked, and not-applicable separately. Missing Metal,
  model access, or an optional dependency is not evidence that its test passed.

See [dependency policy](requirements/README.md) and
[development checks](docs/development.md).

## 11. Performance and MLXBoost

- Profile before optimizing. Identify the measured cost and keep an unchanged
  baseline on the same hardware, software, precision, and workload.
- Record actual input/output counts, warmup, repetitions, timing boundaries,
  batch/concurrency, cache state, synchronization, and completion conditions.
- Separate direct model, full task pipeline, Engine, CLI, and HTTP/SSE timings.
  Do not present an HTTP speedup as a kernel improvement or mix incompatible
  benchmark cohorts. Early EOS uses actual outputs, not the requested budget.
- Distinguish load/conversion peaks, MLX allocation peaks, process RSS, and cache
  bytes; use explicit GB/GiB units. Do not infer parameter count from packed bytes.
- Require repeatable benefit and no new correctness, state, interface, or
  resource regression for promotion. Keep raw runs and failed attempts.
- For a mathematically neutral host optimization, verify affected logits/state,
  tokens, decoded output, lifecycle, and fallback against the unchanged baseline.
  An inherited diagnostic failure need not veto that optimization if it is
  unchanged and disclosed. An unexplained new difference blocks promotion.
- Changes to kernels, precision, reduction order, cache semantics, or forward
  partitioning need the affected numerical/state tests.
- Keep acceleration opt-in and statuses separate from model support. Promotion
  for a tested profile does not automatically enable a default or approve a new
  package. Preserve explicit fallback and rollback.

Use the [benchmark methodology](docs/benchmarks/README.md),
[optimization promotion policy](docs/mlxboost/promotion-policy.md), and
[MLXBoost ownership boundary](docs/architecture/mlxboost-module-boundary.md).

## 12. Storage, downloads, and cleanup

1. Preflight disk space and total memory, including resident weights, execution
   state, KV/cache, conversion temporaries, and download duplication.
2. Run one GPU qualification workload at a time. Download only the exact assets
   required; reuse shared checkpoint bytes for related components deliberately.
3. Put campaign models/caches in a dedicated, ownership-marked work directory.
   Keep evidence outside the directory scheduled for deletion.
4. Save outputs, logs, metrics, identities, and checksums durably before cleanup.
   If persistence fails, report it and secure the evidence before removing weights.
5. Remove task-owned downloaded weights soon after all dependent results are
   saved. Record deleted paths/bytes using portable, publishable references.
6. Delete only paths verified to belong to that campaign. Check resolved paths
   and ownership; never adopt existing user directories or delete an entire
   shared Hugging Face cache as campaign cleanup.
7. Preserve pre-existing user models and assets, including the protected
   `~/models/gpt-oss-20b-MXFP4` and
   `~/mlx-one-models/gpt-oss-20b-MXFP4-Q4` locations from this work.
8. Record interrupted downloads, residual owned files, timeouts, and cleanup
   failures. A checksum mismatch is a failed asset check, not a resumable pass.

The [campaign runner](scripts/qualification/model_release_campaign.py) implements
serial qualification and ownership-based cleanup. The presence of a cleanup
script is not a substitute for checking which paths a particular run owns.

## 13. Evidence and support status

Each admission or claim-changing result must identify:

- Source commit plus relevant source hashes when the worktree differs; runner
  identity, commands, exit codes, timestamp, and environment.
- Checkpoint/revision, asset hashes, conversion, storage/execution precision,
  tokenizer/processor, and task/interface settings.
- Fixture/dataset provenance, actual inputs and outputs, normalization/evaluator,
  expected behavior, and measured results.
- Required checks, diagnostics, failures, skips, excluded scope, memory/storage
  observations, and cleanup disposition.
- Candidate wheel identity when used, differences from the final artifact, and
  the reviewed support or promotion decision.

Use saved machine-readable records and readable summaries. Preserve original
evidence; add a new dated result when behavior or policy changes. Do not relabel a
failed diagnostic as a pass or claim a source smoke was an installed-wheel run.
Exclude secrets, private fixtures, user-specific absolute paths, and model weights
from publishable reports and distributions.

| Situation | Public meaning |
| --- | --- |
| Applicable functional gates passed and reviewed | **SUPPORTED** for the exact documented profile |
| Architecture or tiny tests only | Implemented; real checkpoint support not established |
| Required execution failed or could not run | Failed/blocked for that scope, with the reason |
| Optional experiment or unpromoted extension | Its documented experimental/pending status, separate from the supported primary task |
| Diagnostic missed a quality target | Retain the failed metric; block that quality claim, without inventing a universal support veto |

Keep separate counts for registered model types, supported types, model families,
checkpoint/precision/task profiles, revisions, tests, and modality coverage.
Adding another checkpoint does not necessarily add a model type. Counts must
reconcile with the registry and current support catalog; future growth must not
be blocked by an unexplained hard-coded historical count.

## 14. Documentation and discoverability

- Update the model guide, usage examples, limitations, support status, and release
  notes when their user-visible facts change.
- Keep [README.md](README.md), the packaged support catalog, and GitHub Pages
  consistent. Derive model counts and dashboards from maintained data; use commits
  and changelog entries for development milestones, not invented activity.
- A supported model guide identifies task, checkpoint/precision, installation
  extras/assets, working example, input bounds, evidence, and known limits.
- Document optional experiments separately without applying their status to all
  supported models. Architecture illustrations and roadmap items are not proof
  that a workflow is shipped.
- Generate `README.pypi.md` from the main README. Do not maintain conflicting
  descriptions by hand. Use working absolute source/image links for PyPI.
- Edit maintained website sources, not generated output. Build with strict
  link/anchor checking; preserve existing navigation and visual style.
- Public CI tracking badges are currently omitted. Their absence does not remove
  applicable local validation or disable the existing workflow. Do not restore
  the tracker as part of unrelated development.
- Distinguish local preparation, maintainer approval, repository merge, Pages
  deployment, and PyPI publication. Update publication claims after verification.

See [documentation ownership](docs/documentation-generation.md) and
[site maintenance](website/README.md).

## 15. Licensing and distribution

- Retain Apache-2.0 project licensing and upstream licenses/notices for copied or
  derived code, tokenizer assets, UI components, and conversion sources.
- Check model and dataset terms separately from source licensing. Record gated
  access requirements without placing credentials in source, logs, or commands.
- Keep production dependencies minimal and declared. Review optional-reference
  boundaries when adding a dependency or conversion tool.
- Exclude model weights, raw qualification trees, reference checkouts, caches,
  private data, credentials, local archives, and development environments from
  wheel/sdist payloads.
- Audit exact source, packaged data, UI assets, and notices against the candidate
  artifacts. A successfully built wheel is not by itself a payload audit.

## 16. Git workflow and release handoff

- Work on a focused branch; use an isolated worktree when other work is dirty.
  Preserve unrelated user changes. Review the diff and stage intended files.
- Describe the concrete problem, resulting behavior, validation, limitations, and
  compatibility impact in the pull request. Separate local results from hosted
  checks and distinguish observed causes from suspected CI failures.
- Prepare changes locally for maintainer review. Under the current handoff,
  the maintainer pushes, merges, publishes releases, and uploads PyPI artifacts.
  Do not infer publication authorization from a request to prepare files.
- Keep package metadata, runtime/CLI version, release records, install examples,
  tags, and release notes consistent with the chosen version.
- Build into a fresh candidate directory. Verify metadata, licenses, checksums,
  import boundaries, and clean-wheel behavior outside the source checkout.
- Tie native model results to the actual candidate implementation. When the final
  artifact differs, record and validate the difference; do not copy old hashes.
- Preserve validated artifacts. Rebuilding creates new identities and requires
  refreshed artifact checks; do not replace a recorded wheel silently.
- Review the scope of release automation before invoking it. The legacy
  restricted-alpha validators/workflow do not alone establish the later
  40-model multimodal release. Extend automation explicitly rather than bypassing
  its controls or treating a metadata-only pass as publication approval.

## 17. Acceptance criteria by change type

Every applicable criterion must pass or have an explicit, reviewed disposition.
Use **N/A with a reason** for an unrelated check, not a pretend pass. A missing
required checkpoint/hardware run remains blocked for that claim. This is scoped
acceptance, not a requirement to rerun every historical experiment.

| Change | Required acceptance evidence |
| --- | --- |
| Documentation or image only | Correct facts/links/rendering; generated descriptions synced; strict site build when website output is affected; no model download required |
| Bug fix | Reproduced defect, meaningful regression check, affected functional path restored, related regressions pass |
| New model type | G0–G5, task pack, registry/loading integration, model guide, exact support record, installed package checks |
| New checkpoint, precision, or conversion | New profile identity, loader/contracts, real task result, affected interfaces/regressions, distinct limits |
| Processor/tokenizer change | Checkpoint asset identity, preprocessing/token boundaries, affected real task and interface replay |
| Public API or schema change | Contract/failure-mode tests, working advertised interfaces, lifecycle coverage where affected, compatibility/migration docs |
| Shared cache/scheduler/runtime change | Affected model/state types, isolation, stop/cancel/error cleanup, native regressions, real checkpoint replay appropriate to the changed paths |
| Dependency, Python, MLX, or packaging change | Fresh-environment resolution/import/tests across applicable advertised versions, native replay for backend changes, wheel/sdist and clean-install audit |
| Performance or acceleration | Comparable baseline/change measurements, correctness/state/lifecycle evidence, fallback/rollback, unchanged or explained diagnostics |
| Training or other optional extension | Method-specific correctness, state/save-load/restart tests as applicable, actual workflow evidence, explicit supported/experimental scope |
| Accuracy, parity, context, language, or hardware claim | Predeclared protocol/thresholds, pinned representative fixtures, reproducible measurements within that scope |
| Release candidate | Applicable changes accepted, current support/evidence reconciliation, native regressions, exact artifact checks, consistent docs/version, maintainer decision |

Do not waive a core execution defect to reach a target model count. Conversely,
do not block an unrelated supported model because an optional benchmark or
another modality's extension remains incomplete.

## 18. Standard verification commands

Run from the repository root. Select the blocks appropriate to the change;
integration runs may download only deliberately selected pinned assets. These
commands document the workflow and are not a claim that this document reran it.

### Development environment and ordinary checks

```bash
python3.12 -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" build twine
python -m pip check
python scripts/release/check_lint.py
python -m pytest --junitxml=test-results/pytest.xml
```

Use a separate environment for each other advertised Python version when its
compatibility is affected. Do not reuse packages from another interpreter.

### Native execution on Apple Silicon

```bash
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest --junitxml=test-results/pytest-metal.xml
```

This enables the native test cohort. Real-checkpoint qualification remains a
separate task-specific run with the pinned plan and saved outputs; tiny native
tests alone do not qualify a model. The campaign runner's arguments and ownership
rules are in [its source](scripts/qualification/model_release_campaign.py).

### Documentation and PyPI description

```bash
python scripts/release/pypi_readme.py
python scripts/release/pypi_readme.py --check
python scripts/release/check_documentation.py \
  mlx_one_development_guidelines_v1.0.md CONTRIBUTING.md README.md README.pypi.md
git diff --check
```

For website changes, use a separate documentation environment:

```bash
python3.12 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r website/requirements.txt
python website/build.py
```

### Candidate artifacts

In the development environment, use a fresh directory and audit output for each
candidate; the paths below are examples for a new candidate:

```bash
python scripts/release/pypi_readme.py --check
python -m build --outdir dist/release-candidate
python -m twine check --strict dist/release-candidate/*
python scripts/release/package_audit.py \
  --dist dist/release-candidate --output dist/release-candidate-audit.json
```

Install the exact wheel in a fresh environment and run outside the checkout.
Check `python -m pip check`, CLI version, registry identity, optional import
boundaries, and the relevant installed-task smoke. CLI/registry success alone
does not establish real model execution. Record SHA-256 of the wheel and sdist.
Use the [PyPI handoff guide](docs/releases/0.1.1-pypi.md) for the existing release;
adapt version/path references deliberately for a future one.

## 19. Pull request / release acceptance record

Copy this record into the change description or its evidence summary:

```text
Change / user-visible outcome:
Source commit and relevant source hashes:
Model types / checkpoint revisions / precisions:
Tasks, interfaces, hardware, and input bounds:
Applicable acceptance criteria:
Checks run, commands, and saved result paths:
Results: passed / failed / skipped / blocked / N/A with reasons:
Diagnostics and known limitations:
Regression and compatibility impact:
Artifact identity and clean-install checks, if applicable:
Dataset/model provenance and licensing:
Downloaded assets, cleanup receipt, and protected assets:
README / model guide / support catalog / website updates:
Maintainer decision and remaining publication steps:
```

A change is ready when its promised behavior works, applicable acceptance checks
are satisfied, evidence and limits are reviewable, and documentation describes
the same result. Merge readiness and permission to publish remain separate
decisions.
