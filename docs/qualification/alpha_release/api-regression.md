> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Api regression

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Final source Python/CLI/HTTP completion/completion SSE checks cover ten models; chat/chat SSE cover nine. Saved primary artifacts cover the seven quantized candidates; GPT-2/Gemma primary reuse the identical stored-profile checks. Requests cover deterministic repeats, sampled seeding, decoded stops, usage, finish reasons, malformed/unknown/overflow errors, disconnect, restart and zero handles. Repeated running-cancellation and failed-request recovery are checked on the three minimum targets. Existing queue capacity/authentication tests remain in the full suite. Unexpected 500/SSE exceptions are redacted, with recovery regressions.

[final-checks](../archived-evidence.md); [release-artifacts/wheel-http.json](../archived-evidence.md); [release-artifacts/wheel-primary-M03-http.json](../archived-evidence.md).
