> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Package audit

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

The selected wheel and source distribution were built once from the independent frozen candidate, with the repository’s setuptools/PEP 517 configuration and installed build tools, offline/no isolation. Twine validates metadata. Every production Python file is byte-identical in source, wheel and sdist. The payload audit verifies hashes, notices and exclusion of forbidden/private assets. MLX/Metal are pinned to 0.32.2; the tested dependency constraint file is packaged. Other Python/runtime ranges declared by metadata are not newly certified.

[release-artifacts/package-complete.json](../archived-evidence.md); [release-artifacts/package-audit.json](../archived-evidence.md); [release-artifacts/twine-check.log](../archived-evidence.md).

Every bundled UI file matches the frozen source, and four additional derived-source NOTICE files are included alongside the four MIT notices. Earlier separately frozen builds remain preserved: [artifact lineage](../archived-evidence.md).
