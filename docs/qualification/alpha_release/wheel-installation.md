> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Wheel installation

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

A new venv outside the checkout installs the exact wheel and [server] dependencies offline from cached wheel bytes under the tested constraints; pip itself is updated from an available cached wheel. pip check passes. Ten source model workers assert installed site-packages, real Metal, 29 registrations, no reference packages and zero request handles. TCP HTTP completion/chat/SSE, authentication, error handling, malformed local artifact recovery and shutdown are checked; Qwen3 Q8 is additionally exercised from its exact saved profile.

[release-artifacts/wheel-environment.json](../archived-evidence.md); [release-artifacts/wheel-http.json](../archived-evidence.md); [release-artifacts/wheel-primary-M03-http.json](../archived-evidence.md).

Final runtime/dependency snapshot: [environment-final.json](../archived-evidence.md). Entry environment.json is retained unchanged.
