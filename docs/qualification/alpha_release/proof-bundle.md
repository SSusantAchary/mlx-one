> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Proof bundle

This is a local review bundle for an unpublished **NO-GO** alpha. It preserves failed, preliminary and final executions separately. Final authoritative test counts come from `signoff-tests/default.xml and signoff-tests/native.xml`; earlier runs remain historical observations within this cohort.

[Release decision](release-decision.md), [model matrix](model-support-matrix.md), [release notes](release-notes.md), [automation prerequisites](release-automation.md), [environment](../archived-evidence.md), [source provenance](../archived-evidence.md), [integrity checks](../archived-evidence.md), [complete SHA-256 index](../archived-evidence.md).

Rebuild the derivative reports with `python scripts/qualification/alpha_report.py --run COHORT`, apply the recorded `reproduce-signoff.py` document additions, then index with `python scripts/release/seal_proof.py --run COHORT`. Commands, subprocess exit codes, timestamps, exact revisions, raw numeric metrics and token IDs are retained in the cohort. Production source was frozen before the final test/artifact campaign. Validation of package identity is separate from numerical acceptance. No PASS is granted by report generation. Earlier artifact sets remain preserved and are explicitly superseded in artifact-lineage.json. The selected set includes the final UI and four retained derived-source notices.

Final integrity validation uses the independent candidate checkout because the shared working-tree HEAD remains unchanged. The publication validation must reject this proof. The archive is deterministically sorted, with fixed gzip/tar times, ownership and modes. A local Git bundle carries the review commits; no model weights or reference source trees are added to the proof archive. The referenced checkpoints and reference snapshots remain identified by exact content manifests.
