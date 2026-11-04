# T01 ViT first-divergence investigation

**Decision: backend numerical execution difference for the investigated FP32 fixture (classification B); whole-trace parity remains failed and ViT remains experimental.** The evidence supports accumulated cross-backend numerical differences; it does not identify a particular Metal or CPU kernel as the root cause and does not establish behavior for other inputs. No implementation correction or tolerance change was made.

## Reproduction and boundaries

Transformers CPU eager v4.57.1 matches source commit `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`; the installed modeling file and pinned Git file both hash to `f6497cc352acb8e29c166ca11759bbba79af69c533925a2bdaffb805e8800852`. Checkpoint revision is `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`. FP32 checkpoint file SHA-256 is `1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d`; config and processor hashes match the lockfile. All 200 loaded tensors, including every bias, match source values bit-for-bit after the expected OIHW-to-OHWI patch weight transpose. The source uses epsilon `1e-12`, exact GELU, and two residual sums; the unchanged native encoder uses the same stated operations. Both models were in eval mode, and all traced values and weights are FP32.

The input was the saved `[1,3,224,224]` float32 tensor from the original generated 319×197 RGB image, loaded by both processes; no preprocessing occurred in this diagnostic. The earlier separate processor comparison was bit-identical for that image only. Grayscale, alpha, orientation, other real images, batch sizes and independent task quality are not established. The existing bounded decoder applies EXIF orientation and alpha compositing, which need separate source comparison.

`layer_00` is patch/CLS/position embeddings; `layer_01` is the output of the first transformer block; `layer_05` is the fifth block output (source zero-based `encoder.layer[4]`). `final_hidden` is the output of the final layer normalization after block 12, and logits use the normalized CLS token. Forward-hook and reconstruction checks verify these boundaries. The diagnostic manual MLX trace matches the unchanged production forward exactly on the saved input. Attention scores/probabilities in the native trace are recomputed from Q/K; the fused kernel does not expose its internal scores.

Environment: Apple M4, 32 GiB, macOS 15.8.1 build 24H32, Metal 3, native Python 3.12.11 / MLX 0.32.2 / NumPy 2.5.2; isolated reference Python 3.14.7 / PyTorch 2.15.0 / Transformers 4.57.1 / NumPy 2.5.4. Default `MLX_ENABLE_TF32` was unset. The TF32-off process had `MLX_ENABLE_TF32=0` set before startup. [MLX numerical precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html) describes the option; identical results here do not establish whether this platform would otherwise use reduced-precision matrix operations.

## Before and after layer comparison

No model code was changed, so the original pre-investigation trace and the controlled TF32-off trace reproduce the same elementwise results. The unchanged gate is `abs(error) <= 1e-4 + 1e-4 * abs(reference)`. The normalized error below is the maximum of `abs(error)/(1e-4 + 1e-4*abs(reference))`; values above 1 fail. RMS is absolute RMS error.

| Boundary | Before max abs | Controlled max abs | Normalized error | RMS | Failed elements |
|---|---:|---:|---:|---:|---:|
| layer_00 | 1.621246e-05 | 1.621246e-05 | 0.050252 | 1.2866e-06 | 0 |
| layer_01 | 1.716614e-05 | 1.716614e-05 | 0.10706 | 2.0397e-06 | 0 |
| layer_02 | 1.66893e-05 | 1.66893e-05 | 0.10672 | 2.5178e-06 | 0 |
| layer_03 | 1.692772e-05 | 1.692772e-05 | 0.12238 | 2.7892e-06 | 0 |
| layer_04 | 3.623962e-05 | 3.623962e-05 | 0.17754 | 3.2714e-06 | 0 |
| layer_05 | 0.0007324219 | 0.0007324219 | 1.865 | 6.8957e-06 | 1 |
| layer_06 | 0.003631592 | 0.003631592 | 1.4298 | 1.9236e-05 | 1 |
| layer_07 | 0.004455566 | 0.004455566 | 0.91052 | 2.3967e-05 | 0 |
| layer_08 | 0.004516602 | 0.004516602 | 0.74727 | 2.4984e-05 | 0 |
| layer_09 | 0.004516602 | 0.004516602 | 0.88849 | 2.5863e-05 | 0 |
| layer_10 | 0.004638672 | 0.004638672 | 1.019 | 2.8325e-05 | 1 |
| layer_11 | 0.004577637 | 0.004577637 | 2.4395 | 3.793e-05 | 62 |
| layer_12 | 0.004394531 | 0.004394531 | 2.7103 | 5.5036e-05 | 168 |
| final_hidden | 9.918213e-05 | 9.918213e-05 | 0.42592 | 5.7977e-06 | 0 |
| logits | 4.959106e-05 | 4.959106e-05 | 0.097446 | 4.7565e-06 | 0 |
| probabilities | 1.078844e-05 | 1.078844e-05 | 0.070006 | 3.5412e-07 | 0 |

## Block 5 operator trace

The chained column follows the full native model; independent replay uses the exact reference input to each operation. Both residual-addition replays are bit-identical. Block 4 was also traced: all independent operations pass, and its chained output has maximum normalized error 0.17754. Block 5 driven by the exact reference block input passes at every captured boundary, including its final residual (max absolute error 0.0002213; normalized error 0.0876).

| Operation | Shape | Chained max abs | Chained max normalized | Independent max abs | Independent max normalized |
|---|---|---:|---:|---:|---:|
| input | 1×197×768 | 3.62396e-05 | 0.17754 | — | — |
| ln1 | 1×197×768 | 2.62447e-06 | 0.025509 | 4.76837e-07 | 0.0014149 |
| query | 1×197×768 | 1.12057e-05 | 0.060364 | 4.05312e-06 | 0.014658 |
| key | 1×197×768 | 1.04904e-05 | 0.052913 | 5.24521e-06 | 0.013936 |
| value | 1×197×768 | 6.38515e-06 | 0.058163 | 3.8147e-06 | 0.014301 |
| scores | 1×12×197×197 | 2.28882e-05 | 0.10538 | 0 | 0 |
| probabilities | 1×12×197×197 | 3.21865e-06 | 0.018352 | 3.57628e-07 | 0.0019149 |
| attention_output | 1×197×768 | 8.46386e-06 | 0.061983 | 3.8147e-06 | 0.019692 |
| residual1 | 1×197×768 | 3.62396e-05 | 0.23372 | 0 | 0 |
| ln2 | 1×197×768 | 4.41447e-06 | 0.042053 | 3.57628e-07 | 0.0017107 |
| mlp1 | 1×197×3072 | 4.673e-05 | 0.24592 | 5.72205e-06 | 0.011691 |
| gelu | 1×197×3072 | 4.57764e-05 | 0.17058 | 5.96046e-07 | 0.0051358 |
| mlp2 | 1×197×768 | 0.000701904 | 0.53392 | 0.000190735 | 0.05359 |
| residual2 | 1×197×768 | 0.000732422 | 1.865 | 0 | 0 |

All listed shapes are float32 in both backends. Detailed shape, dtype, RMS, gate and per-element failure counts for blocks 4 and 5, including direct replay and whole-block isolation, are in `summary.json` in the local evidence directory. Independent replay bypasses accumulated input error; it does not prove equivalence on every possible input.

## One-change execution experiments

Each variant changes one diagnostic choice at a time from TF32-off baseline. Native CPU uses MLX CPU rather than Metal. Trace runtimes include instrumentation and were not benchmarked as production inference; performance implications are NOT MEASURED.

| Variant | Block 5 failed elements | Block 12 failed elements | Max logits error | Finding |
|---|---:|---:|---:|---|
| TF32 off, fused attention | 1 | 168 | 4.95911e-05 | baseline |
| Default environment | 1 | 168 | 4.95911e-05 | bit-identical to TF32 off on this input |
| Unfused attention | 1 | 158 | 5.14984e-05 | still fails; not a production change |
| Approximate GELU | 142238 | 149531 | 0.043191 | not source-equivalent; worsens error |
| Explicit source-order mean/variance | 1 | 166 | 5.14984e-05 | still fails |
| Contiguous transposed linear weights | 1 | 168 | 4.95911e-05 | bit-identical to baseline |
| Reverse both residual add operands | 1 | 168 | 4.95911e-05 | bit-identical to baseline |
| Materialize FP32 intermediates | 1 | 168 | 4.95911e-05 | bit-identical to baseline |
| MLX CPU | 1 | 3 | 7.62939e-06 | different errors; still fails |

TF32 default and disabled traces are bit-identical in every shared recorded array. Attention eager uses explicit matmul, scaling and precise softmax; the native fused scores themselves are inaccessible. The explicit normalization ablation uses the same epsilon but a separately expressed reduction. Reversing addition operands reproduces the source order; all values remain FP32, and forcing intermediate materialization is a scheduling/rounding diagnostic, not evidence about hidden hardware accumulator precision.

## Regression and limits

The native-task, opt-in Metal TrOCR/OCR and Qwen VLM tests passed: 77 passed, two dependency deprecation warnings (`MLX_ONE_RUN_MLX_TESTS=1`). The full default test suite passed: 2,205 passed, 79 skipped, nine passing subtests and two dependency warnings. The full suite uses the current pre-existing dirty checkout and is not a clean-baseline comparison. No shared OCR or classifier implementation was edited: their pre/post SHA-256 values remained `7cb215dccdecc8d70b3e91832bf841185c575954cd185f36357af5c911d17b97` and `c4d79b2c892f8ec55f40f00c96fc48fe11d38c3b6658d3e8cdd724ab48ef4adb`, respectively. MLXBoost code, flags and evidence were not edited.

Raw traces, saved input, complete comparison JSON, weight audit, full-suite log, exact command and environment manifest are under ignored `local/native_tasks/runs/vit-investigation-final-20261102/`. The new diagnostic scripts create outputs exclusively and do not run in production inference. Repeated evidence is not a substitute for real-image task quality, processor edge cases, B2 validation, CLI, HTTP, packaging or release qualification. The next bounded step is to investigate sensitivity of the first failing element across multiple fixed real and generated inputs and establish a source-faithful correction only if a specific implementation defect is demonstrated.
