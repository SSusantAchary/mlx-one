# P0.1 external comparison

No oMLX performance win is established. The current matched environment has
MLX 0.32.2, MLX-LM 0.31.3 and NumPy 2.5.2; oMLX is not installed.

| Comparator | Pinned identity | Status | Reason |
|---|---|---|---|
| Standalone MLX-LM | 0.31.3 / `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd` | FAIL (Qwen3 128) | Same-checkpoint numerical gate failed; performance blocked |
| oMLX stock | `03bc3dd7118a5f1bbf9cb9ac119f551c77477b09` | BLOCKED | Not installed; pinned MLX 0.32.4 and NumPy <2.4 differ from this environment |
| oMLX optimized | same pin | BLOCKED | Same dependency barrier; actual patch engagement unmeasured |

Do not upgrade the qualification environment to manufacture a comparison.
Historical compatible releases are separate comparator versions, not evidence
for the current oMLX pin. External engines are benchmark comparators only and
are never imported as mlx-one runtime backends.

## Source and license inspection

The uncommitted source receipts are under
`qualification/results/p01_math/reference-inspection/source-pins.json`. Every
available snapshot has an exact revision, path, SHA256 and source URL.

* [oMLX affine projection routing, lines 140–256](https://github.com/jundot/omlx/blob/03bc3dd7118a5f1bbf9cb9ac119f551c77477b09/omlx/patches/qwen35_q4_mlp.py#L140): shape and per-call-row gating are design guidance, not copied thresholds or kernels.
* [oMLX GDN dispatch, lines 35–124](https://github.com/jundot/omlx/blob/03bc3dd7118a5f1bbf9cb9ac119f551c77477b09/omlx/patches/qwen35_gdn_chunked.py#L35): exact dimensions, mask/state exclusions and actual route engagement matter. Its M5 results do not validate M4 execution.
* [oMLX D256 dispatch, lines 207–240](https://github.com/jundot/omlx/blob/03bc3dd7118a5f1bbf9cb9ac119f551c77477b09/omlx/patches/qwen35_fa256_attention.py#L207): hardware and launch-budget gates are retained as principles; no native extension is vendored.
* [MLX 0.32.2 fast bindings](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/python/src/fast.cpp): inspect the installed API before assuming a gated-delta primitive exists.

oMLX's inspected LICENSE is Apache-2.0. MLX and MLX-LM's inspected licenses are
MIT. Top-level NOTICE lookup returned unavailable at these pins; this is an
inspection outcome, not an assertion that no other notices exist. No reference
engine source or custom kernels are copied into the native runtime.

## Comparison protocol

Use exact offline assets and rendered token IDs, native affine packed weights,
FP16 floating execution/KV, B1, no prefix reuse, paging, SSD or speculation.
Separate prepared-history forced-token forwards from Engine generation and
HTTP service timing. Match source/environment, checkpoint, context, seeds and
fixtures; unsupported or mismatched comparisons fail closed.

Artifacts require numerical proof, ten measured repeats, two independent
processes, supervised balanced ABBA order, complete memory telemetry and a
declared protected workload matrix. Installing a patch is insufficient: actual
optimized calls must be observed. The comparator does not promote registry
capabilities.

A named-profile win requires both prefill and decode >=1.10x the competitor,
peak MLX memory <=1.05x, correctness and reliability passes, and no protected
throughput regression >3%. Missing conditions produce PENDING, measured losses
produce FAIL. No universal claim or default change follows from one profile.

## Executed Phase A comparison

Frozen runtime source digest:
`921448a4b6a6d05c1e742cb1c2fa7309192863f414db58ee43f2dd63089896b2`.
The 128-token rendered prompt and 128-token forced continuation were matched
exactly, with native INT8 payloads and explicit FP16 floating execution/KV.
Native reference/head-vector parity passed four ABBA workers; a fresh two-worker
parity cohort also checksummed the numerical NPZ payload after a demonstrated
evidence-writer defect was repaired. Historical incomplete checksums were not
silently upgraded into new qualification evidence.

Standalone MLX-LM failed the unchanged FP16 numerical gate on all 129 logits
vectors, with one greedy mismatch and maximum absolute error 0.109375.
No comparator performance measurements ran after this failure. This observation
does not by itself identify which arithmetic implementation causes the difference.
Artifact: `qualification/results/p01_math/phase-a-mlx-lm/1792241823109029000`.
The earlier checksum failure and both numerical cohorts remain local evidence.

The native prepared-history diagnostic completed two workers per route, with
one measured repetition each. Those counts are below the ten-repeat gate and
cannot establish a material improvement or competitor win. Engine generation,
forced-forward timing and client streaming timing remain separate workloads.
