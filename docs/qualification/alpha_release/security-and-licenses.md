> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Security and licenses

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Unexpected HTTP/SSE errors return a generic 500 message; sentinel path/credential tests verify redaction and recovery. The malformed safetensors clean-wheel probe verifies rejection and subsequent valid load. Authentication, queue admission and disconnect cleanup retain their tested contracts. The certificate bundle, DOMPurify, devalue and Vite were patched to the fixed versions documented by the recorded advisories, and the static UI rebuilt/tested. Full vendor notices now accompany bundled UI code. Apache-2.0 project licensing and four MIT source notices are present in both distributions. No local references, model weights, historical reports, credentials or machine paths are bundled.

[dependency-advisory-details.json](../archived-evidence.md); [release-artifacts/dependency-audit.json](../archived-evidence.md); [ui-security-followup.json](../archived-evidence.md); [ui-npm-audit-2.json](../archived-evidence.md); [release-artifacts/package-audit.json](../archived-evidence.md).

Final runtime/dependency snapshot: [environment-final.json](../archived-evidence.md). Entry environment.json is retained unchanged.
