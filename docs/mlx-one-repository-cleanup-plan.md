> Executed on 2026-10-22. See the [cleanup execution report](repository-cleanup-report.md). The original audit and recommendations below are retained for context.

# MLX-One — Repository Cleanup Plan for 0.1.0 Alpha

**Snapshot audited:** 2026-10-21 14:50 IST (48,576 file entries, including ignored files).  
**Local repository:** `/Users/susant-achary/dev_work/git_codes/c_base/mlx-one`  
**Scope:** recommendations only. **No files were removed from the user's Mac or GitHub.**  
**Evidence limitation:** a filesystem inventory does not show Git tracking, dependency/import usage, generated-file policies, disk sizes or whether files have changed since the snapshot. Verify these before acting.

## A. Delete regenerable artifacts from disk (only when no active process uses them)

| Exact path / pattern | Inventory count | Action |
|---|---:|---|
| `.pytest_cache/` | 6 | Delete |
| `.ruff_cache/` | 89 | Delete |
| `build/` | 358 | Delete (regenerate on package build) |
| `dist/` | 2 | Delete stale wheels, rebuild for release |
| `src/mlx_one.egg-info/` | 6 | Delete |
| `ui/node_modules/` | 11,967 | Optional disk cleanup; reinstall with `npm ci` |
| `ui/.svelte-kit/` | 62 | Delete |
| `ui/test-results/` | 1 | Delete |
| `**/__pycache__/`, `**/*.pyc` | many | Delete, never commit |
| `**/.DS_Store`, `.coverage` | multiple | Delete |
| `examples/text-sft/predictions-smoke.json` | 1 | Move to local test output if it is regenerated, not an asserted baseline |

**Keep:** `.git/` (never delete), `.venv/` if currently used (ignore it; rebuild only voluntarily), all actual source code and legitimate test fixtures, and `src/mlx_one/ui/dist/` until Python wheel packaging/release checks prove it can be regenerated as part of the build.

## B. Keep local but remove from Git tracking / never publish as source

Recommended in-repo ignored workspace (or sibling folder `../mlx-one-private/`):

```text
local/
  prompts/
  plans/
  reports/
  evidence/
  experiments/
  assets-drafts/
```

### Existing top-level folders: keep locally but exclude from Git

| Path | Files in snapshot | Recommendation |
|---|---:|---|
| `.venv/` | 14,497 | Local installation only, already ignored |
| `references_libs/` | 6,854 | Preserve both `mlx-lm-main/` and `transformers-main/` locally; exclude from Git, not MLX-One source |
| `qualification/results/` | 5,015 | Archive run receipts and raw parity outputs locally (or as external CI/release artifact), not in source Git |
| `benchmarks/results/` | 1,325 | Local/generated; current `.gitignore` already ignores `benchmarks/results/p1/` only |
| `runs/` | 96 | Local/generated; already ignored |
| `calibration/results/` | 50 | Raw host-specific results local; keep a small, accurate public summary in `docs/calibration.md` or `compatibility/baselines/` |

### Generated research evidence within docs: move to `local/evidence/` as *whole self-contained bundles*

- `docs/qualification/alpha_release/proof-20261021-1792544161047727000/` (1,068 files)
- `docs/qualification/alpha_release/numerical-correction-20261021-1792560186243989000/` (1,243 files)
- `docs/qualification/alpha_release/proof-20261021-1792544161047727000.git.bundle`
- `docs/qualification/alpha_release/proof-20261021-1792544161047727000.tar.gz`
- Associated `proof-20261021-1792544161047727000-*` signature, checksum, archive descriptor, and reproduction files; preserve them together and recheck hashes/relative paths
- `docs/qualification/ten-models/reference-first-20261020-1792502347743318000/` (604 files)
- `docs/qualification/m4-32gb/ten-model-qualification-20261020-1792475268786913000/` (472 files)
- `docs/qualification/m4-32gb/numerical-correction-20261020-1792496951619201000/` (382 files)
- `docs/qualification/m4-32gb/revamp-20261020/` (31 files)
- `docs/inference_optimization/baseline/` (33 files)
- `docs/inference_optimization/phase1_repair/` (22 files)
- `docs/inference_optimization/tiled_feasibility/` (75 files)
- `docs/inference_optimization/qwen_causal/` (32 files; audit non-generated reference/code contents before moving)

**Caution:** proof bundles may have internal relative paths and integrity manifests. Moving them can break scripts that assume a particular parent directory. Preserve a byte-for-byte archive, record SHA256, and check any release/evidence references before staging removal. Do not simply delete the only copy.

### Prompts: move to `local/prompts/`

- `mlx-one Native Web UI Implementation Prompt.md`
- `p1_hardware_qualification_prompt.md`
- `p1_reference_guided_coding_prompt.md`
- `docs/alpha_0_1/MLX_One_0.1.0_Alpha_A0-A10_Unified_Codex_Prompt_v5.md`

### Plans and development drafts: move to `local/plans/`

- `cache.md` (unless current canonical documentation)
- `mlx_one_LTS.md`
- `mlx_one_development.md`
- `mlx_one_full.md`
- `mlx_one_p01_native_math_implementation_test_plan.md`
- `mlx_structure.md`
- `mlxone_inference.md` (once canonical inference doc is chosen)
- `mlxone_roadmap.md` (preserve curated public `docs/roadmap.md`)
- `p0_implementation_plan.md`
- `phase_0_1.md`
- `folder_structure.txt`
- `PROJECT_STRUCTURE.md` (if auto-generated development inventory)
- `docs/architecture/README-before-proof-release.md`
- `docs/migration/` (4 historical audit/migration documents; archive privately after implementation references are updated)
- `docs/qualification/alpha_release/numerical-policy-proposal.md` (historical proposal, not public support policy)

### Reports and local machine data: move to `local/reports/` or `local/evidence/`

- `recent_report.md`
- `updated_review_cleanup.md`
- `num_error.md`
- `model_performance.md`
- `model_list.txt`
- `scripts/qualification/checkpoint-paths.local.json` (machine-specific; keep under existing location if runner depends on it, and add precise ignore rule)
- `docs/alpha_0_1/inventory.md`
- `docs/alpha_0_1/progress.md`
- `docs/alpha_0_1/resume-release-blockers.md`
- `docs/alpha_0_1/baseline.md`
- `docs/alpha_0_1/execution-report.md`
- `docs/alpha_0_1/release-inclusion.md`
- `docs/alpha_0_1/qwen3-mlx-lm-parity.md` (private deep research, optionally publish a summary)
- `docs/alpha_0_1/qwen3-upstream-native-adoption.md` (private engineering study unless intended as developer documentation)
- `docs/qualification/ten-models/changed-files.md`, `mathematical-port-diff.md`, `numerical-root-cause-analysis.md`, `remaining-reference-first-blockers.md` (retain local study / link concise public status)
- `docs/qualification/alpha_release/assertion-audit-20261021/` (signed detailed audit; preserve local)

## C. Keep publicly, and consolidate rather than delete

### Root: retain

- `README.md`, `LICENSE`, `CITATION.bib`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`
- `pyproject.toml`, `MANIFEST.in`, `CHANGELOG.md`, `.gitattributes`, `.gitignore`, `.github/`
- `src/`, `tests/`, `scripts/`, `ui/src/`, `ui/static/`, `ui/package.json`, `ui/package-lock.json`, Svelte/Vite/TypeScript/Playwright configs
- `benchmarks/*.py`, `benchmarks/model_parity/*.py`, `qualification/*.py`, `examples/`, `compatibility/`
- All relevant `UPSTREAM_LICENSE`, `NOTICE`, source provenance and attribution files; do not discard upstream license notices

### Public-facing docs (keep accurate, compact)

- `docs/alpha_0_1/installation.md`, `public-api.md`, `model-capabilities.md`, `precision-support.md`, `support-matrix.md`, `support-matrix.json`, `release-notes.md`, `experimental-boundary.md`, `qualification.md`, `source-ownership.md`
- `docs/api/openai-compatibility.md`
- `docs/model-catalog.md`, `docs/roadmap.md`, `docs/cache-runtime.md`, `docs/calibration.md`, `docs/ui.md`
- `docs/architecture/mlx-one-v0.1.0-revamp.md` (make it a current architecture document, not an in-progress plan)
- Curated summaries from `docs/qualification/alpha_release/`: `known-limitations.md`, `model-support-matrix.md`, `release-scope.md`, `hardware-qualification.md`, `security-and-licenses.md`, `wheel-installation.md`, `release-notes.md` (if accurate)
- Curated summaries from `docs/qualification/ten-models/`: `final-10-model-matrix.md`, `final-results.md`, `reproduction.md` (if still accurate)
- `docs/qualification/first-10-models.md` (keep if it describes supported candidates accurately; don't imply successful full qualification)

**Root document normalization:** The GitHub README currently references `mlx-one_inference_engine.md`. If it is the active overview, move it to `docs/architecture/inference-engine.md` only after updating README, internal references, doc links and any tests. Do not remove it first.

**Assets:** Keep `mlx-one.png`, `mlx-one-stack.png`, `mlx-one_arch.png`, and `mlx_One_chat_ui.png` until README/docs references are updated. Relocate used assets to `docs/assets/` as a separate doc-only commit. Move `Metallic Silver Button with Bold M.png` to `local/assets-drafts/` if it is only a logo draft.

**Ship the UI:** `ui/src/` is frontend source; `src/mlx_one/ui/dist/` is the 19-file, potentially packaged static app. Do not remove bundled assets unless wheel-install smoke tests prove the release packaging builds them independently.

**Experimental MLX code:** Keep `src/mlx_one/engine/`, `src/mlx_one/models/{language,audio,embeddings,vision_language,segmentation}/` and native adapters in source. Move implementations under an experimental namespace *only* as a separate compatibility-preserving refactor after import, API, and test checks; do not delete because a checkpoint is not alpha-qualified.

## D. `.gitignore` additions and corrections

The public `.gitignore` (checked 2026-10-21) already ignores `.venv/`, `build/`, `dist/`, `runs/`, Python caches, `node_modules/`, `.svelte-kit/`, `test-results/`, and `benchmarks/results/p1/`. It does **not** broadly exclude `qualification/results/`, `calibration/results/`, or `references_libs/`. It also contains an unnecessary `.gitignore` entry on its first line; remove that self-ignoring entry.

Add, merging with existing rules (never overwrite blindly):

```gitignore
# Developer-only context
/local/
/references_libs/
/scripts/qualification/checkpoint-paths.local.json

# Machine-generated experimental evidence
/qualification/results/
/benchmarks/results/
/calibration/results/

# Frozen/private historical artifact directories (after archival)
/docs/qualification/alpha_release/proof-20261021-*/
/docs/qualification/alpha_release/numerical-correction-20261021-*/
/docs/qualification/alpha_release/proof-20261021-*.git.bundle
/docs/qualification/alpha_release/proof-20261021-*.tar.gz
/docs/qualification/ten-models/reference-first-20261020-*/
/docs/qualification/m4-32gb/ten-model-qualification-20261020-*/
/docs/qualification/m4-32gb/numerical-correction-20261020-*/
/docs/qualification/m4-32gb/revamp-20261020/

# Tool-generated local coverage / macOS metadata
.DS_Store
```

If the historical `docs/inference_optimization` evidence is also moved to `local/evidence/`, ignore only its old evidence directories or the new `local/` path; retain authored public docs there.

**Important Git behavior:** Ignore rules do not untrack already-committed files.

## E. Safe execution order (run on the Mac, not on the uploaded inventory)

1. `git status --short`, `git branch --show-current` and `git ls-files > ../mlx-one-tracked-before.txt`. Save any uncommitted changes. Work on a new `chore/alpha-repository-cleanup` branch.
2. Inventory cross-references before relocation: `rg -n 'qualification/results|calibration/results|references_libs|docs/qualification|mlx-one_inference_engine|checkpoint-paths.local' README.md pyproject.toml MANIFEST.in .github scripts src tests docs`.
3. Copy and verify signed proof and raw qualification evidence to a separate private archive, or a local ignored folder. Check checksums and relative references. Keep at least one preserved copy.
4. Move explicitly listed prompts/plans/reports to ignored `local/`, keeping historical information available locally. Move only exact known files, no broad `rm` of docs.
5. Merge `.gitignore` updates and update links in README/docs, tests, build metadata and packaging.
6. Remove only **Git tracking** from evidence paths that are confirmed safe to omit from a clean clone (example below). This leaves files on disk. Then re-run CI on a fresh checkout because ignored local files can conceal missing fixtures.

```bash
# From repository root; review the staged deletion list before any commit.
git rm -r --cached --ignore-unmatch -- qualification/results calibration/results benchmarks/results

git status --short
git diff --cached --stat
```

7. Verify `python -m pytest`, release script checks, `python -m build`, wheel install and API import/CLI/HTTP smoke checks in clean environments, plus `npm ci && npm run build` and available UI tests. Inspect `pyproject.toml` / `MANIFEST.in` because `src/mlx_one/ui/dist/` may be wheel data. Compare model support claims with tested evidence. Only then commit.
8. After safe cleanup, delete only confirmed *regenerable* caches; do not use `git clean -fdx` or recursive deletion of `.git`, `src`, `tests`, `references_libs` or raw evidence before backup.

## F. Suggested final repository shape

```text
mlx-one/
├── .github/
├── benchmarks/           # Reproducible runners and model parity code; no raw results
├── compatibility/        # Compact device-baseline / compatibility summaries
├── docs/
│   ├── alpha_0_1/        # Published installation, APIs, support matrix, limitations
│   ├── api/
│   ├── architecture/
│   ├── assets/
│   └── qualification/     # Curated, truthful results and how to reproduce
├── examples/
├── qualification/        # Executable harness code only; no results/
├── scripts/
├── src/mlx_one/          # All supported/experimental modality implementations
├── tests/
├── ui/                   # SvelteKit source and package manifest, no node_modules
├── README.md
├── CHANGELOG.md
├── LICENSE
├── MANIFEST.in
└── pyproject.toml

# Ignored on each developer's machine, present only when desired
local/{prompts,plans,reports,evidence,experiments,assets-drafts}/
references_libs/{mlx-lm-main,transformers-main}/
.venv/
```

## G. Validation gate

- No import/API/CLI/HTTP/SSE regression.
- Fresh wheel includes the UI bundle and required model license notices.
- `git ls-files` returns no private prompt drafts, local checkpoint paths, raw benchmark receipts, locally vendored upstream repositories, virtualenv files or node_modules.
- Public support claims are backed by retained high-level summaries and do not turn failed/unexecuted checkpoint tests into claims of support.
- Archived proofs remain reproducible from their preserved private archive (or external release artifact), with hashes retained.
- Github README images/document links still resolve.

**Outcome:** a clean *source repository*, with reproducible tools and transparent alpha status, while retaining the complete local research and validation history for future work.
