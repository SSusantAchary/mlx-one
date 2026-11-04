# Measured bottleneck and scope

The unmodified original protocol was reproduced in
`local/mlxboost/runs/baseline-original-20261101/`: 27/27 isolated server sessions
passed, with six timed requests per model/runtime. MLX-One medians were
20.81/19.99/14.65 E2E tok/s for Qwen2.5/Qwen3/SmolLM2.

The separate native diagnostic used three timed requests in one process per
model, with synchronized boundaries. Direct forward including greedy selection
measured 131.94/107.85/24.81 tok/s; bare native generation measured
125.65/106.24/24.63. Device-owned managed generation measured
109.06/96.11/24.16. These are native diagnostics, not HTTP numbers. Keeping the
whole iterator on the device owner deliberately removes outer thread handoffs.
Raw token times and device-owner profiles are in `native-prechange-20261101/`.

The HTTP profile in `http-profile-20261101/qwen2-1-mlx-one/profiles/` identified:

1. `pressure.execution_forecast` reconstructs a full qualification scope on every
   token when the registry is non-None. Serving supplies a registry even without
   activation calibration. `selection.source_identity` recursively reads and
   hashes every Python source file. `native_report` inventories modules, tensor
   dtypes and execution contracts; `host_identity` launches sysctl twice.
2. Native evaluation and admission/accounting are the next measurable costs.
   The existing generator uses one synchronized logits evaluation followed by
   argmax extraction; it does not recompute the full prefix. The existing dense
   cache already retains capacity in 256-token steps.
3. Remaining serving handoffs and token delivery require further isolated
   attribution after eliminating the impossible lookup. No second optimization
   has been implemented or mixed into this experiment.

In the instrumented aggregate, 256 token forwards spent 15.389 s inside
`execution_forecast`, including 8.645 s in source identity and 5.231 s in
`native_report`; total profile wall time was 20.897 s. Inclusive times overlap.
Profiling materially slowed requests (about 12.3 tok/s), so these numbers are
attribution evidence, not uninstrumented latency predictions or GPU timings.
The uninstrumented flag ablation is the causal performance evidence.

The improvement's upper bound is the removable lookup, not all forward time.
The shortcut still enumerates live registry records every forward; any
`activation_peak` candidate restores the complete existing scope validation and
fallback. Memory forecasts and admission continue on every token. No stale
identity, calibration or negative result is cached across requests.

Checkpoint packed metadata was read from actual safetensors headers and retained
in `audit-20261101/checkpoint-shapes.json`. Qwen2.5 Q projection is packed
`[896,224]` U32, gate `[4864,224]`; Qwen3 Q projection `[2048,256]`, gate
`[3072,256]`, affine Q8/group64. Existing layers use stock `nn.QuantizedLinear`.
No repeated full-weight dequantization was found. There is no kernel-level GPU
capture, thermal sensor record, power record or stock-MLX bottleneck evidence.
Custom Metal, new SDPA, quantized wrappers and compile experiments therefore did
not pass their entry gate.
