> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Release automation

The workflow at `.github/workflows/release.yml` is prepared and statically checked locally; **it has not run on GitHub**. It separates static checks, backend-independent tests, native hardware tests, one distribution build with isolated native wheel/HTTP smoke tests, candidate approval validation and optional publication of the same checksum-verified artifacts.

The self-hosted runner must carry the labels `self-hosted, macOS, ARM64, mlx-one-hardware`, have real Metal access, Python 3.12.11, enough disk/memory, and the exact checkpoint assets at the paths recorded in the selected evidence manifest. Hardware jobs serialize in `mlx-one-m4-qualification`. The smoke installation is a fresh venv outside the source checkout, with reference imports forbidden. No Linux job supplies GPU qualification evidence. The local isolated source/primary inference and TCP HTTP evidence is linked from [wheel-installation](wheel-installation.md).

Before any publication request, the owner must approve a defensible numerical policy and scope, resolve failing mandatory gates and lint disposition, approve the exact release commit, and configure required reviewers for the `pypi-manual-approval` environment plus the matching PyPI Trusted Publisher (repository, workflow, environment). These remote settings were not changed here. No permanent publishing token is stored in source. Index availability must be rechecked; the recorded PyPI and TestPyPI project lookups returned 404, which is not proof of every historical publication.

The workflow has no push/tag publication trigger. `publish` defaults to false. Candidate validation requires GO, approved numerical contract, approved owner decision, complete mandatory gates, the approved minimum model count, source/evidence hashes and a clean tracked checkout. **This NO-GO proof is expected to reject publication.** TestPyPI/production uploads and release tags remain unperformed.

Integrity verification: `python scripts/release/validate.py --root CANDIDATE --evidence COHORT`. Add `--publication` to enforce the approval gate. Frozen source is `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`. See [proof-integrity.json](../archived-evidence.md) and [evidence-index.json](../archived-evidence.md).
