# Prepared integration pull request

Title: **Integrate native task APIs and verified documentation for 0.1.1**

Base: `main`, verified at `eb8bd3887ace3217ba23dc14525cef9c8a77e50d`.
Head: `integration/native-tasks-publication-20261104`.
Status: local branch prepared; no authenticated GitHub session was available.
No PR, release, tag or merge was published automatically.

## PR body

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

## Exact commands for an authenticated environment

Run from the isolated integration checkout. Authentication is required before remote publication; maintainer review is
required before merge. These commands intentionally create a
draft PR and do not merge it:

```sh
git switch integration/native-tasks-publication-20261104
git fetch origin main
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
# Compare origin/main with the verified base; update and retest if it changed.
git diff --exit-code eb8bd3887ace3217ba23dc14525cef9c8a77e50d origin/main
gh auth status
git push --set-upstream origin integration/native-tasks-publication-20261104
gh pr create --draft --repo SSusantAchary/mlx-one \
  --base main --head integration/native-tasks-publication-20261104 \
  --title 'Integrate native task APIs and verified documentation for 0.1.1' \
  --body-file docs/integration/2026-11-04/pr-body.md
```

The prepared patch and Git bundle remain in ignored
`local/publication-integration/runs/20261104/`. Their hashes and final branch
identity are recorded in `handoff.json` there. The bundle includes all integrated
commits relative to the frozen base, without unrelated working-tree changes.
