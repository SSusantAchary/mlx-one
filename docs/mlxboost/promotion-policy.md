# Numerical diagnostics and optimization promotion policy

Policy clarification: 2026-11-02. This clarification supersedes the blanket
interpretation that any strict numerical failure blocks every optimization.

Record two independent decisions for each tested checkpoint, precision, device
and workload:

- **Model qualification:** run the strict pinned-source and cached/fresh gates
  with the existing thresholds. Retain failures, raw evidence and failing
  diagnostic exit codes. A failed model gate remains FAIL and prevents claiming
  that model is numerically qualified.
- **Optimization promotion:** evaluate the change against the unchanged baseline
  in the same execution configuration. A pre-existing model failure does not
  automatically block a mathematically neutral optimization that introduces no
  numerical, state, output or functional regression and has a repeatable measured
  end-to-end benefit.

Mathematical neutrality must be supported by code inspection and measured
reference/flag-off/flag-on comparisons. Removing redundant host metadata work is
an example. Preserve model operations and their order, operands, shapes, dtypes,
quantization, cache semantics, sampling and detokenization. Compare captured
logits/state bit for bit, exact token IDs and decoded bytes, EOS, finish reason
and usage. Check affected API behavior, cancellation, memory accounting and
fallback. Text agreement alone does not establish neutrality. Scope the decision
to the configurations and boundaries actually tested.

For an inherited failure, retain before/after results from the same strict
diagnostic and show that the optimization does not introduce or worsen it. An
unexplained difference blocks promotion through this neutral-change route.
Changes to kernels, precision, reduction order, forward partitioning or cache
behavior require the affected numerical/state qualification; mathematical
equivalence on paper is insufficient. Profiling remains required before custom
Metal work.

Source provenance, native-only packaging, API compatibility, independent feature
flags, tested rollback and reproducible performance evidence remain required.
Keep optimization status (`PROMOTE`, `EXPERIMENTAL`, `REJECT`, `BLOCKED`) separate
from model-qualification status. `PROMOTE` applies only to the documented tested
envelope and does not automatically enable a default or qualify an alpha release.

## Caveat note

**The strict numerical gate remains a diagnostic and model-qualification tool.
An inherited failure is not an automatic veto on a demonstrated mathematically
neutral performance optimization.** Such an optimization may be promoted for its
measured scope while the model's numerical qualification remains failed. Report
the unresolved failure beside the performance result; neither a speedup nor
unchanged generated text proves general numerical correctness, long-context
quality or qualification of other checkpoints/devices.

For the current `fast_decode` host-lookup shortcut, retained comparisons show
bit-identical captured logits/KV and exact native outputs, alongside unchanged
cached/fresh failures in native and pinned source. Those inherited failures no
longer disqualify independent optimization promotion. The existing opt-in
`EXPERIMENTAL` deployment status and defaults are unchanged by this policy edit;
this edit makes no new measurement or rollout decision. See the
[qualification matrix](qualification-matrix.md),
[change comparisons](optimization-ablation.md) and
[strict numerical evidence](numerical-parity.md).
