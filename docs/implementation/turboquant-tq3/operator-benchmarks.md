# Reproducing operator measurements

Run from the repository root, with native MLX-One installed in the candidate environment and the
pinned reference installed in a separate development environment. Each invocation is a separate
process; never leave the full model loaded in another server while comparing backends.

```sh
.venv/bin/python benchmarks/turboquant/operators.py --backend native --profile gpt \
  --rows 1,2,8,16,32,64,128,256 --repetitions 12 --output /tmp/tq3-native-operators.json
/tmp/mlx-one-tq3/reference-env/bin/python benchmarks/turboquant/operators.py \
  --backend reference --profile gpt --rows 1,2,8,16,32,64,128,256 \
  --repetitions 12 --output /tmp/tq3-reference-operators.json
.venv/bin/python benchmarks/turboquant/compare.py --native /tmp/tq3-native-operators.json \
  --reference /tmp/tq3-reference-operators.json --output /tmp/tq3-operator-comparison.json
.venv/bin/python benchmarks/turboquant/dispatch_sweep.py \
  --reference-root /tmp/mlx-one-tq3/reference --repetitions 12 \
  --output /tmp/tq3-dispatch-sweep.json
```

`--profile small` is a bounded quick check. `--operations dense,expert,router_expert,moe,decoder`
selects cases. GPT profile uses actual K/O attention shapes and 32 expert banks with top four;
blocks use the real native/reference layer classes, synthetic FP16 parameters and full causal
attention without KV. Checkpoint BF16 mix and generation/cache/serving are separate workloads.

Seeds, packed fixture construction, codebooks, signs, inputs and geometry match across backends.
Reports record host/device, MLX version, reference SHA, shape, per-sample timings, median, p95,
effective projection GFLOP/s, first-call cost, graph-build median, active baseline, peak active
increment and peak process RSS. First call includes compilation and is recorded separately;
three warmups precede the twelve synchronized samples. Graph-build cost includes Python dispatch
and lazy allocation setup, not an isolated malloc benchmark. Peak increment includes rotation,
routing and output buffers, not just shader scratch. Matrix FLOP throughput is not defined for
whole blocks; use their latency. No GPU hardware allocation-event counter is claimed.

`mx.eval` plus `mx.synchronize` completes GPU work before each stop time. Timed wall clock includes
normal operator graph construction and dispatch. Layer-level profiling in `full_model.py` is a
separate synchronized pass and must not be mixed into steady-state throughput.

Retained reports: baseline before tuning, dispatch sweep, final native/reference runs, per-case
comparison and the rejected shared-input experiment. Full operator comparisons are synthetic and
must never be promoted to full-model decode or prefill parity.
