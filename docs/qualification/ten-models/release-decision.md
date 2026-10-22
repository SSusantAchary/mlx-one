> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Alpha release decision

**NO-GO.**

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

Actual production work replaces mathematics for six families and verifies/reuses two already matching families. Cross-reference continuation/state parity improves from 2/10 entry checkpoints to 9/10. Seven of eight families pass strict whole-port equivalence with independent component controls. Qwen3.5 retains its justified precision fixes and still fails full-model cross gates. All ten checkpoints and all primary profiles execute.

Source-qualified: 0/10. Primary-qualified: 0/10. Fully qualified: 0/10. All references still fail canonical self-consistency; diagnostics never override these gates. Strict task-format/model-quality and quantization losses remain separate. APIs pass 58/58 applicable source combinations.

1822 passed, 54 skipped, 3 warnings in 63.12s (0:01:03). Native acceptance 353 passed, 2 warnings in 7.70s. Entire native 8 failed, 2066 passed, 18 skipped, 3 warnings in 99.55s (0:01:39). Eight inherited failures, zero newly failing identities; Ruff zero new findings. Fresh-wheel all-ten-model and HTTP/chat/SSE smoke passes without reference libraries. Hardware measurements are bounded characterization only.

[Complete matrix](final-10-model-matrix.md), [remaining blockers and reproduction](../archived-evidence.md), [executed engineering summary](../archived-evidence.md). Historical reports and both reference trees remain preserved. No commit, push, source-library download, Engine replacement or optimized backend promotion was performed.
