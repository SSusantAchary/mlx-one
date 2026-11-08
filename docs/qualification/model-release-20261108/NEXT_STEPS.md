# Release handoff

**Maintainer approved 0.1.1 for release on 8 November 2026. Publication is pending.**
See the [release notes](../../releases/0.1.1.md) and
[release decision](../../releases/0.1.1.json).

All **40 selected model types** have passing native primary-workflow records under the [MLX-LM-aligned support policy](SUPPORT_POLICY.json). See [current technical checks](RELEASE_CANDIDATE.md) and the [per-model catalog](../../../src/mlx_one/data/model_support_release_20261108.json).

The model implementation work and evidence are prepared locally. The branch is `release/model-qualification-40-20261108`; no commit, push, tag, website deployment or package publication has been performed by this campaign. Existing frozen-source and publication controls still apply when publishing; they do not change the model support count.

Optional future work follows the claims being made: additional checkpoint sizes or quantizations, longer contexts, multilingual speech, document regions/layout, broad accuracy benchmarks and other hardware. These are not prerequisites for the supported primary workflows. Actual failed loading, invalid outputs or interface defects must still be resolved before advertising the affected operation.

Historical quality failures and independent reference comparisons are preserved. [Resolutions](RESOLUTIONS.md) identifies repairs and diagnostic limits; the [historical rollup](README.md) and `retests.json` retain the stricter original protocol. Do not replace failed raw results with passing benchmark results.

Campaign-owned checkpoint downloads have been removed after saving results. Both protected user GPT-OSS folders were retained. See `cleanup-resolution-audit.json` for the final scan and native asset hashes.
