# Documentation PR preparation

Title: **docs: publish scoped native-task results and MLXBoost module boundary**

No remote branch or GitHub PR was published. GitHub CLI and authenticated
GitHub connector/token credentials were unavailable. The prepared local branch
is `docs/readme-publication-20261104`, isolated from the owner's dirty worktree.
Its changes contain documentation only; the main worktree index is preserved.

## Description

Publish four bounded experimental task profiles with checkpoint revisions,
measured task quality, historical latency, interface scope and explicit
qualification failures. Preserve the existing README artwork, quickstarts and
model-performance table. Document MLXBoost as an optional default-off internal
module, with standalone extraction proposed only.

The evidence map binds claims to local sealed artifacts and existing reports.
The audit distinguishes fresh regression/wheel verification from historical
performance. No model code, optimization, threshold or dependency changes are
included.

Validation: 2,238 default tests passed, 79 skipped and nine subtests passed;
122 focused tests and nine subtests passed. Generated text/VLM/audio/OCR checks
passed. All 1,379 sealed native artifacts, 44 reference source files, checkpoint
assets and 46 protected-file hashes verify. Clean-wheel and link-check details,
including any external access failures, are in [verification.json](verification.json).
The default and focused selections overlap and are not summed.

## Merge prerequisite

The audited native task and MLXBoost implementations already exist in the
owner's dirty development tree and are outside this documentation patch.
The original base commit alone does not contain them. Apply these docs on top
of that implementation; do not merge a docs-only branch in a way that advertises
unavailable task APIs. The README explicitly identifies results as belonging
to the local development wheel, not a published package release.

## Exact patches

The ignored directory `local/readme-publication/runs/20261104/` contains:

- `publication.incremental.patch`: README changes relative to the captured
  pre-audit dirty README, plus newly authored publication documents and ADR.
  Apply to the audited dirty tree, not directly to the old base commit.
- `publication.base.patch`: complete documentation diff on the isolated branch
  against base `c0bcfba3b0d19cd50a6b547ab170e89592083438`, including the exact
  existing task/MLXBoost evidence documents needed by the new links.
- `patch-manifest.json`: patch hashes, file inventories and base identity.

Review from the repository root before applying on a separate checkout:

```sh
git apply --stat local/readme-publication/runs/20261104/publication.base.patch
git apply --check local/readme-publication/runs/20261104/publication.base.patch
```

The check must run against a checkout of the stated base with the patch file
accessible there; it is expected to conflict with already-present documentation
in the audited dirty tree. The incremental patch's README preimage SHA-256 is
`af606fb9d94faca029fc27b18aed82212d8a7b5ee076a7ed6cb663603570419c`.

After implementing the merge prerequisite and authenticating, a maintainer can
push the prepared documentation commit/branch and open a draft PR with the
title/description above. Review the actual remote base before publishing.
