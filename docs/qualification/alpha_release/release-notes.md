> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Release notes

Proposed MLX-One 0.1.0a1; **NO-GO / unpublished**. No checkpoint is alpha-certified.

This candidate retains eight native model families and all 29 architecture registrations. It distinguishes a registered architecture from a qualified exact checkpoint in runtime/model metadata; Qwen3.5 remains EXPERIMENTAL_NOT_QUALIFIED. It redacts unexpected HTTP/SSE errors, fixes eight inherited test fixtures without XFAILs, patches vulnerable certificate/UI dependencies, and packages retained source/UI licenses with a pinned MLX 0.32.2 runtime.

Executed qualification: 1,834 default tests pass (67 skipped), 2,099 native tests pass (18 skipped), 25 unchanged upstream cache tests pass. Nine candidate checkpoints pass semantic cache audits in source and primary profiles. Original frozen full/cache qualification remains 0/10; the unapproved proposed contract qualifies 0/9. A trained Gemma BF16 full-query attention witness fails the independent FP64 gate. Numerical qualification, policy approval and lint-debt disposition prevent release.

Source and saved-primary APIs pass all applicable 58 and 52 cells. The immutable 30-prompt fixture shows no native-only graded task regression for the nine candidates; shared model failures remain recorded. Clean isolated wheel inference runs without mlx-lm or transformers. All results are bounded to the recorded M4/32GB/Python 3.12.11 environment.

[Decision](release-decision.md), [exact model matrix](model-support-matrix.md), [raw gate summary](../archived-evidence.md).
