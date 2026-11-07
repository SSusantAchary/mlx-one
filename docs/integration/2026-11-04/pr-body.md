Commit the previously local native task implementations with the README/ADR
publication work, so users can install the APIs described by the documentation.
The candidate reports version 0.1.1 throughout Python, distribution metadata,
CLI and OpenAPI. Four task profiles remain experimental: SST-2 sentiment,
BERT NER, CLIP 1–10 labels and mono16k Silero VAD.

Keep existing model mathematics, numerical gates and MLXBoost defaults unchanged.
MLXBoost stays optional inside MLX-One; standalone extraction remains proposed.
Frozen benchmark/qualification evidence is retained. Failed candidates remain
blocked, and no GPT-OSS provenance or SAM3 raw-archive qualification is added.

Validation: 2,238 default tests passed, 79 skipped and nine subtests; 160 native
selection tests passed, 17 skipped. The selections overlap. The 0.1.1 native-only
wheel passes 1,566 tensor comparisons across 17 real-checkpoint replay cases,
bounded task-quality evaluations, Python/CLI/HTTP identity and four authenticated
loopback-server checks. All 46 protected hashes remain unchanged. See
[merge readiness](merge-readiness.md), [verification](verification.json) and the
[complete changed-file inventory](changed-files.json).

The original dirty worktree and its index are preserved. Unrelated dirty files
are excluded. Pinned checkpoints, evaluation text, media and reference archives
remain local and are not bundled in the wheel or Git changes. Dataset/license,
blocked-model and wider device/task qualification limitations remain explicit.
