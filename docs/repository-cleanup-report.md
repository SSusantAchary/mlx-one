# Repository cleanup execution — 2026-10-22

Executed the [repository cleanup plan](mlx-one-repository-cleanup-plan.md) on branch
`chore/alpha-repository-cleanup`. Existing uncommitted implementation work was preserved;
no commit or push was made in the user's repository. Temporary Git commits exist only in the
isolated validation snapshot under `/tmp/`.

## Changes

* Archived private prompts, plans, reports, raw proof/evidence bundles and six machine-specific JSON
  receipts under ignored `local/`. Relocations retain original repository-relative paths.
* Removed Git tracking from **150 generated result files**, plus the archived root
  `model_performance.md`. These 151 staged deletions remove tracking only; the data remains local.
* Preserved both upstream reference trees locally and excluded `references_libs/`, all generated
  result directories, local checkpoint paths and `local/` from source control. Removed the
  self-ignoring `.gitignore` rule. Packaging explicitly prunes private/local calibration data.
* Moved the inference overview to `docs/architecture/inference-engine.md`, updated the architecture
  document, fixed public links and consolidated support matrices against the existing 13-candidate
  manifest. No support status, checkpoint validation, experimental feature or release approval was
  promoted. Historical numerical decisions remain archived observations, separate from current policy.
* Preserved the Parakeet configuration/723-shape contract as a small explicit test fixture.
  Ordinary tests no longer require raw result receipts. Only optional external-reference comparisons
  skip when upstream checkouts are absent; native-only arithmetic/cache tests still execute.
* Preserved `examples/text-sft/predictions-smoke.json`: it is a supplied scoring input, not generated
  output. All runtime Python source files match the pre-cleanup SHA-256 inventory.
* Fixed UI regeneration to retain the original third-party license notice through
  `ui/static/THIRD_PARTY_NOTICES.txt`. The notice was recovered byte-for-byte from the previously
  verified wheel and is now copied automatically by the frontend build. The packaged UI remains.
* Removed regenerable tool/Python caches, stale root build/dist outputs and egg metadata after checks.
  `.venv/`, `ui/node_modules/`, references, raw results, source, tests, fixtures and bundled UI remain.

## Preservation

The ignored receipt directory `local/evidence/cleanup-20261022/` contains the original Git status,
tracked-file inventory, binary patches, source snapshot, complete relocation archive, per-file
checksums, move ledger, commands/logs and package audit. The complete relocation archive includes
nested frozen build/dist artifacts, including wheels and signed-proof sidecars.

* Complete relocation archive: **4065 files**;
  SHA-256 `363e9ebd3386ee6ee11fcd2bd505a2e892568e2cce21f0fde945beb502e55f3a`.
* Six additional machine-data files: SHA-256 `8649d75e94f942279bb878ec89cb5ff52979b97a56f1ae78c24fd669f3af1fd3`.
* Three original evidence indexes rechecked **1067 file hashes** with
  **zero differences** after relocation. Signed payload bytes were not edited or re-signed.
* Cache cleanup removed **75 generated paths**, totaling
  **20.19 MiB logical file sizes. Archival copies intentionally
  consume additional local space; this operation does not claim large disk-space savings.

See the [archive/reproduction policy](qualification/archived-evidence.md). Frozen proofs must be
reproduced with their original source/policy and required host assets, not silently revalidated
against the current changed source. No full model checkpoint was downloaded by cleanup.

## Validation

| Check | Result |
|---|---|
| Full repository default suite | 1979 passed, 68 skipped |
| Full repository Metal suite, optional references available | 2245 passed, 18 skipped |
| Isolated source default suite, ignored/private data absent | 1979 passed, 68 skipped |
| Isolated source Metal suite, upstream references absent | 2212 passed, 51 skipped |
| Ruff debt gate | 65 preexisting findings; zero new |
| UI `npm ci`, type/Svelte check | PASS; 0 errors, 0 warnings |
| UI unit tests | 7 passed |
| UI browser tests | 3 passed |
| UI production rebuild | PASS; bundled notice retained |
| Clean source wheel and sdist source/UI/license/private-payload audit | PASS |
| Fresh wheel dependency check | PASS |
| Fresh wheel native Python generation | PASS using existing cached GPT-2 |
| Fresh wheel real HTTP/chat/SSE/auth/error recovery | PASS |
| Fresh wheel TQ3 Python/CLI/API/SSE/cancel/unload | PASS using fabricated tiny GPT |
| Public README/docs link audit | 0 broken local links |
| Git tracked private paths/raw results/reference trees | None |

The isolated snapshot has no `local/`, upstream checkouts, generated result trees or machine-data
receipts. Twelve newly optional upstream comparison cases skip there; existing optional-reference
skips account for the rest of the Metal count difference. All 2245 cases pass on this host with the
preserved reference checkouts. The temporary source snapshot has its own test-only Git revision so
inventory tests can capture provenance normally. Initial UI-notice and snapshot-setup failures are
retained in private logs and were resolved before these final results.

Package audits compare all production Python and UI bytes, required MIT/Apache/TQ3/UI notices,
and forbidden/private payload rules. Current alpha publication remains **UNAPPROVED/PENDING**;
large MoE checkpoints retain their prior architecture/backend-only statuses. Operator and historical
small-checkpoint results have not been turned into full-model performance or quality claims.

## Review and recovery

Only intentional removal of tracking was staged. Other existing and cleanup edits remain reviewable
in the working tree; unrelated implementation changes were not committed or reset. Review with
`git status --short` and `git diff --cached --stat`. To stop ignoring restored optional data, edit the
specific ignore rule deliberately. To recover pre-cleanup source/docs, extract the source snapshot
into a separate scratch directory and overlay the complete relocation archive and optional data.
Keep private evidence out of source-control commits and future distribution archives.
