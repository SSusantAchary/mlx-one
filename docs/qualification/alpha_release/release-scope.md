> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Release scope

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Proposed version `0.1.0a1`; restricted text alpha target M01/M03/M09, minimum three checkpoints. Apple M4 32 GB, macOS 15.7.9, Python 3.12.11, MLX/Metal 0.32.2. Native MLX, B1 reference_dense, one startup model, FP16 attention KV. Gemma/Phi use sliding state; LFM convolution history follows its activation dtype. No checkpoint is certified. Qwen3.5 remains experimental. Continuous batching, paging, INT8 paged KV, speculation, SSD persistence and unqualified contexts/conversions are outside this scope.

[release-manifest.json](../archived-evidence.md); [index-version-audit.json](../archived-evidence.md). The indexes returned 404 at the recorded check; version availability must be rechecked at publication.

Final runtime/dependency snapshot: [environment-final.json](../archived-evidence.md). Entry environment.json is retained unchanged.
