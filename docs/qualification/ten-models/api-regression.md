> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Public API regression

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

| Interface | Source checkpoints passed / applicable |
|---|---|
| python | 10/10 |
| cli | 10/10 |
| http_completion | 10/10 |
| completion_sse | 10/10 |
| chat | 9/9 |
| chat_sse | 9/9 |

All 58 applicable source checkpoint/interface combinations pass after final cache-version integration. Raw checks include deterministic greedy output, usage, finish reasons, decoded stops, malformed/unknown/overflow errors, streaming assembly, cancellation/disconnect, restart and zero active handles/reservations. Existing one-startup-model residency remains unchanged. Derived in-memory profiles pass Python/HTTP applicable checks; their eight CLI selector cases remain UNSUPPORTED. The two saved quantized artifact CLI/API checks pass separately.

The final wheel is installed in a new environment from offline cached dependency wheels. All ten pinned source models execute from site-packages. HTTP completion/chat/SSE and one-byte Unicode wire decoding pass. MLX-LM and Transformers are absent, and an import guard rejects either reference library. All 29 registrations remain. [Fresh wheel results](../archived-evidence.md).
