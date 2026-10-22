# 0.1.0 alpha candidate — unreleased draft

Authorized local Qwen3-0.6B affine Q8 B1 workflow now passes Python, CLI, HTTP/SSE and lifecycle checks on M4. Quality coverage, a native floating-weight cell and approved-source wheel installation remain pending, so no checkpoint is ALPHA_SUPPORTED. The A10 worker now records measured lifecycle and explicit missing coverage. Details: [native validation](native-qwen3-validation.md).

Project name, logo and message retained. Version remains 0.1.0a1; no publication or new support claim.

Native implementations and existing text/vision/audio/retrieval/SAM3/training/export/evaluation/compatibility/UI interfaces remain in place. Changes harden finite-weight and quantization validation, text shard containment, embedding pipeline/default context behavior, VLM pixel/EXIF preprocessing and bounded HTTP audio decode. Public exports/registered architectures preserved; OpenAPI now uses actual alpha version. NumPy runtime dependency is explicit, and ordinary pytest resolves existing test helpers.

All five precision targets and four initial modality tracks are described separately from checkpoint support. No current-source checkpoint is ALPHA_SUPPORTED yet. Historical M4 receipts, numerical failures and experimental optimizations remain preserved and unpromoted. New offline examples are API-validated only. HTTP audio is bounded to 25 MiB / 300s with a 120s decoder timeout; malformed/non-finite inputs reject. Python long-form file decoding remains available.

Release NO-GO until core text and real floating/affine cell gates, explicit release inclusion, wheel/sdist content audit and clean installed-wheel usability pass. ASR is preferred additional qualification; optional embedding/VLM pending does not independently block a narrower text release.
