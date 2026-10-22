# Archived research and release evidence

Raw run receipts, signed proof bundles, wheels, private prompts and development reports are retained
locally and excluded from the source repository. Their historical outcomes and signatures were not
rewritten. This is an archival change, not a model-support promotion.

Current support is defined by the [model admission policy](../model-admission.md),
[model support matrix](alpha_release/model-support-matrix.md) and packaged
`src/mlx_one/data/compatibility/alpha_release.json`. The compact
[functional evidence summary](functional-evidence-summary.json) identifies the original proof and
raw-record SHA-256. The former E/KL/FP64 research gates are not ordinary functional admission gates.

## Preserved local layout

`local/evidence/repository/` preserves original repository-relative paths for complete relocated
bundles: alpha-release proof and numerical-correction cohorts (including signatures, archives and
reproduction sidecars), ten-model/M4 cohorts, inference-optimization captures, and native-MoE/TQ3
raw validation evidence. Source provenance and license notices remain public. Upstream checkouts
stay at ignored `references_libs/`; generated results stay at their original ignored result paths.
These directories are optional developer data and are absent from a clean source checkout.

The cleanup receipt is `local/evidence/cleanup-20261022/`:

* `working-tree-before.tar`: pre-cleanup authored source and documentation snapshot.
* `relocated-bundles.tar`: complete original paths and bytes for all relocated payloads, including
  nested frozen `build`/`dist` artifacts. 4065 files verified.
* Complete archive SHA-256: `363e9ebd3386ee6ee11fcd2bd505a2e892568e2cce21f0fde945beb502e55f3a`.
* `relocated-files-sha256.json`, `backup-files.json`, `moves.json`, original Git status and binary
  patches record per-file integrity, exact moves and preexisting uncommitted work.

For historical reproduction, extract the source snapshot into a separate scratch directory, then
extract the complete relocation archive over it. Original relative paths are restored. Copy the
needed preserved raw result/reference directories into that scratch layout, or use the frozen Git
bundle/source reproduction scripts supplied with the proof. Run historical validators against their
original source and policy, not the current changed source/support matrix. Machine-specific absolute
paths may require the original host or explicit runner arguments. No portable requalification or
signature re-signing is claimed by relocation.

The original reference/source snapshots, signed files and internal relative paths remain unchanged.
No archive is uploaded or published automatically. Restoring a private proof is optional for ordinary
unit tests, native tests, UI development and wheel installation.

## Optional checkpoint manifests and host receipts

The former public checkpoint manifest, environment/import audit, and three native-MoE smoke
receipts contain host-specific paths and now live under `local/evidence/repository/` at their
original relative locations. A separate `additional-machine-data.tar` preserves all six files;
SHA-256: `8649d75e94f942279bb878ec89cb5ff52979b97a56f1ae78c24fd669f3af1fd3`. They are absent from a clean source checkout.

For a new host, copy `scripts/release/checkpoint-manifest.example.json` to an ignored local file
and set each `path` to its pinned downloaded checkpoint. Pass that file explicitly via
`scripts/release/wheel_smoke.py --assets FILE --model M01 [--http] --output NEW.json`.
The retained original manifest is at
`local/evidence/repository/docs/qualification/alpha_release/checkpoint-manifest.json`.
No full checkpoint is downloaded by repository cleanup.
