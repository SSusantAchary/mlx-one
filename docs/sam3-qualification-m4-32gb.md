# SAM3 M4/32 GB qualification — partial results

Recorded: 2026-10-02. Native execution used `/Users/susant-achary/venvs/mlx/bin/python`
on the Apple M4/32 GB host, outside the sandbox so Metal could initialize.
This is **partial qualification**, not completion of the SAM3 qualification plan.
The package continues to report `qualified=False`.

## Checkpoint and reference contract

| Item | Inspected / used value |
| --- | --- |
| Checkpoint | `facebook/sam3`, composite `sam3_video`, float32 safetensors |
| Checkpoint revision | `3c879f39826c281e95690f02c7821c4de09afae7` |
| Reference | Transformers `6133195dcb027feb7502379acd13750e167eb6c5` (`5.3.0.dev0`) |
| Native environment | Python 3.13, MLX 0.32.2, NumPy 2.2.6, Pillow 11.3.0 |
| Isolated reference environment | Python 3.12, PyTorch 2.7.1, torchvision 0.22.1; pinned Transformers source |
| Process budget | 24 GiB (`25,769,803,776` bytes), out of 32 GiB unified memory |
| Image fixture | `hf-internal-testing/sam2-fixtures/truck.jpg`, RGB, 1800 × 1200 |
| Fixture SHA-256 | `941715e721c8864324a1425b445ea4dde0498b995c45ddce0141a58971c6ff99` |

Strict native loading passed for the detector (1,468 tensors), interactive model
(685 tensors), and composite video model (1,797 tensors). These are load-contract
checks, not task-quality evidence. No quantization or reduced precision was used.

The reference uses unmodified pinned neural-network inference math. Optional
CUDA-only bookkeeping kernels are supplied by NumPy mask-IoU NMS and
eight-connected components, shared with the native validation code. This kernel
policy is part of the comparison scope; it is not a comparison against CUDA kernels.
PyTorch and Transformers are validation dependencies, not production imports.

## 32-frame video results

The controlled clip consists of the fixture rolled horizontally by `index % 8`
pixels over 32 frames. The prompt is `truck`. It contains one retained object;
it is **not natural video** and does not exercise occlusion or new arrivals.
Offline and streaming were captured and compared separately, with one fixed
identity mapping: reference ID `0` → native ID `0`, never rematched per frame.

| Mode | Frames | Retained detections / IDs | Minimum mask IoU | Maximum score difference | Maximum box difference | Fixture gate |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Streaming | 32 | One object, ID `0`, every frame | 0.99998276 | 0.00007511 | 0 pixels | PASS |
| Offline | 32 | One object, ID `0`, every frame | 0.99998275 | 0.00007511 | 0 pixels | PASS |

All retained detections match. Every compared mask meets IoU ≥ 0.98, every score
difference is ≤ 0.02, and every pixel-space XYXY box differs by ≤ 2 pixels.
No identity swap was observed in this single-object fixture.

Per-frame measurements and capture checksums are recorded in the
[streaming comparison](../qualification/results/sam3-streaming-truck-32-m4.json)
and [offline comparison](../qualification/results/sam3-offline-truck-32-m4.json).
The large `.npz` captures remain local validation artifacts, not repository files.

| Native mode | Total session seconds | Peak MLX bytes | Peak process resident bytes |
| --- | ---: | ---: | ---: |
| Streaming | 150.9784 | 8,407,588,697 | 5,759,238,144 |
| Offline | 248.5289 | 8,399,855,453 | 6,700,122,112 |

Both reported peaks are below the 24 GiB budget. Peak MLX allocation and process
RSS are separate measurements and must not be added together on unified memory.
Timings include processing/postprocessing and are not isolated throughput benchmarks;
other validation work was running concurrently. Reference CPU session totals were
879.2671 seconds (streaming) and 879.2987 seconds (offline), with peak process RSS
7,312,818,176 and 8,191,885,312 bytes respectively. Do not infer a speedup from these runs.

The streaming metadata fix retains the chronological frame count while evicting
processed frame tensors. Offline originals are retained on temporary disk for replay,
and temporal histories use host storage. The sessions exited their context managers.
Network-free small-model tests cover cleanup, cancellation, isolation, and failure
paths; a measured repeated-real-session allocation-baseline test is **still pending**.
The final opt-in native checkpoint/lifecycle/comparison suite passed `29 passed`.
The selected backend-free SAM3/inspection/schema/CLI regression suite passed
`70 passed, 7 skipped`. Documentation links and Python examples checked successfully;
`git diff --check` passed. These checks do not substitute for the pending real-scene
qualification matrix or full repository regression suite.

### Preserved failed comparison

The preliminary PyTorch **MPS** reference comparison failed: reference frame 0
retained zero masks while native retained one. Frames 1–31 matched the numeric gates
(minimum IoU approximately 0.99335, maximum score difference 0.00320, box difference
at most 2 pixels), but the missing detection means the overall MPS comparison is
**FAIL**, not a pass with frame 0 excluded. The cause is not established.
The subsequent full CPU reference retains one object in every frame and passes
both 32-frame comparisons. CPU and MPS results are not interchangeable evidence.
See the [failed MPS comparison](../qualification/results/sam3-streaming-truck-32-mps-reference.json).

## Image fixture checks

| Workflow | Scope | Result |
| --- | --- | --- |
| Concept | `truck`, one retained mask, initial image preprocessing | PASS: IoU 0.9996150, score difference 0.0002104, box difference 0.32664 pixels |
| Interactive | One positive center point, three candidate masks, corrected preprocessing | PASS: minimum IoU 0.9950716, maximum predicted-IoU difference 0.0010584 |

The initial Pillow resize path failed interactive parity on two alternative masks
(IoU approximately 0.762 and 0.660; predicted-IoU difference approximately 0.0677).
After matching antialiased bilinear preprocessing, all three masks meet the gates.
Do not treat the earlier failed capture as a pass. Concept confidence and interactive
predicted-IoU scores are different quantities.

## Reproduction

Activate the requested environment and use an authorized local snapshot:

```bash
source ~/venvs/mlx/bin/activate
python benchmarks/sam3_qualification.py \
  --model /path/to/pinned-sam3-snapshot --task video \
  --image /path/to/truck.jpg --text truck --frames 32 --streaming \
  --output /path/to/new-stream-native.npz
```

Omit `--streaming` for the independently measured offline mode. Use an isolated
reference environment with the pinned Transformers source on `PYTHONPATH`, the
same arguments, and `--backend reference --reference-device cpu` to capture
the reference. Do not use the installed production environment's Transformers
version as an equivalent reference.

```bash
python benchmarks/sam3_compare.py \
  --native /path/to/new-stream-native.npz \
  --reference /path/to/new-stream-reference.npz --frames 32 \
  --output /path/to/new-comparison.json
```

Capture and comparison outputs refuse overwriting. The comparison exits nonzero
for missing frames, mismatched detections/IDs, or any numeric gate failure.

## Remaining gates — unqualified

- Concept positive/negative box exemplars; current-preprocessing concept recapture.
- Interactive box prompts and refinement parity; automatic-mask pipeline parity.
- Natural ≥32-frame clips with multiple objects, occlusion, new arrivals, corrections,
  removal, and retained-detection matching with one fixed ID map per clip/mode.
- Real-checkpoint repeated-session baseline, cancellation during inference, and
  shared-model session cleanup under the 24 GiB budget.
- Encoded-video decoding/export qualification and full repository regression coverage.

These results must not promote the entire SAM3 family to qualified support.
See the [model guide](../src/mlx_one/models/segmentation/sam3/README.md) for current
APIs and restrictions. Streaming intentionally omits future-dependent hotstart
removal, as described by the [upstream video reference](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video).
