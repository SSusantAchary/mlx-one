# SAM3 image segmentation and video tracking

Native float32 MLX inference for the original
[SAM3 checkpoint](https://huggingface.co/facebook/sam3). Registered types:
`sam3` (concept detector), `sam3_tracker` (interactive image segmentation),
`sam3_tracker_video` (temporal tracker architecture), and `sam3_video`
(composite detector/tracker). The detector includes the ViT feature pyramid,
CLIP text encoder, geometry encoder, DETR decoder, presence scoring, and masks.
The interactive path uses point/box/mask prompts and a two-way decoder. The
composite video path shares the image backbone and adds temporal memory,
association, reconditioning, and object lifecycle state.

**Partial qualification only.** The pinned checkpoint passes the controlled
32-frame single-object CPU-reference comparisons separately for offline and
streaming on M4/32 GB. This is not natural-video or multi-object qualification.
See [measured results and remaining gates](../../../../../docs/sam3-qualification-m4-32gb.md).
Standalone `sam3_tracker_video` is architecture-only: use the composite checkpoint
for the public video API. Training, text generation, HTTP/UI integration,
quantization, reduced-precision defaults, and SAM3.1 are unsupported.

## Installation and assets

On Apple Silicon, install the existing Pillow vision extra:

```bash
python -m pip install -e '.[vision]'
hf auth login
```

Request and accept authorized access at the publisher's checkpoint page first.
Do not put access tokens in scripts or prompt JSON. The checkpoint license is
separate from mlx-one's Apache-2.0 license; see [notices](NOTICE.md).
Loading uses safe `.safetensors`, config/processor JSON, and tokenizer assets;
pickle `.pt` weights and remote code are not loaded. The default revision for
`facebook/sam3` is `3c879f39826c281e95690f02c7821c4de09afae7`.
An offline directory must contain a complete compatible snapshot, not only weights.

FFmpeg and ffprobe on `PATH` are needed only for encoded local video files.
Frame directories and Python RGB frame iterators need no video decoder. NumPy
inputs must be `uint8` RGB arrays of shape `(height, width, 3)`; paths and PIL
images are also accepted. Frame dimensions must stay fixed within a session.

## Image API

Examples below show available APIs, not qualification of every prompt.
Coordinates are original-image pixels; boxes are XYXY and point labels are
`1` (positive) or `0` (negative). Use concept box labels separately from point labels.

```python
from mlx_one.segmentation import load_segmentation_model, segment_image

model = load_segmentation_model("facebook/sam3", task="concept")
result = segment_image(model, "image.jpg", text="person")
print(result.scores, result.boxes, result.masks.shape)
```

Concept segmentation also accepts `boxes=[[x0, y0, x1, y1], ...]` and
`box_labels=[1, 0, ...]` for positive/negative exemplars. These prompt combinations
still need real-checkpoint parity qualification.

```python
from mlx_one.segmentation import load_segmentation_model, predict_masks, generate_masks

model = load_segmentation_model("facebook/sam3", task="interactive")
result = predict_masks(model, "image.jpg", points=[[100, 120]], point_labels=[1])
best = int(result.predicted_iou.argmax())
refined = predict_masks(
    model, "image.jpg", points=[[100, 120]], point_labels=[1],
    mask=result.low_res_logits[best], multimask=False,
)
automatic = generate_masks(model, "image.jpg", points_per_side=32)
```

Interactive calls accept one box per prompt group, and refinement accepts finite
2-D mask logits, not concept confidence. Automatic generation currently uses a
single-image point grid, predicted-IoU/stability filtering, and mask-IoU NMS.
It does not implement crop-layer/box-NMS automatic generation parity; it remains
unqualified. Large grids retain CPU candidate masks and can be expensive.

All results contain original-resolution boolean masks, pixel XYXY boxes, original
dimensions, and checkpoint provenance. Concept confidence (`scores`) is distinct
from interactive `predicted_iou`; low-resolution logits support refinement.

## Video API

```python
from mlx_one.segmentation import load_segmentation_model

model = load_segmentation_model("facebook/sam3", task="video", offline=True)
with model.video_session("frames/") as session:
    prompt_id = session.add_prompt("person")
    for result in session.propagate():
        print(result.frame_index, result.object_ids, result.prompt_ids)
```

`frames/` contains lexically ordered images; use zero-padded filenames. A local
video path or an iterator of RGB arrays can replace it. Offline sessions retain
original frames on temporary disk for corrections and repropagation.
`correct_object(object_id, frame_index, points=..., point_labels=..., box=...,
mask=..., prompt_id=...)` targets an object; new objects need an existing concept
`prompt_id`. A video correction mask is an original-resolution boolean array.
Object removal, prompt removal, reset, close, and cancel are session-owned operations.

For chronological Python streams, use `model.video_session(streaming=True)` and
`session.process_frame(frame)` per frame. Streaming rejects past-frame corrections
and omits future-dependent hotstart removal; qualify it separately from offline
propagation. CPU histories and bounded GPU feature retention are separate from
text dense/APC/block caches. Exceeding the process budget raises a resource-limit
error rather than changing tracking settings or silently dropping objects.
Context exit releases session state without unloading another session's shared model.

## CLI and exports

```bash
mlx-one segment facebook/sam3 image.jpg --mode concept --text person --output results-concept
mlx-one segment facebook/sam3 image.jpg --mode interactive --prompts-json prompts.json --output results-interactive
mlx-one segment facebook/sam3 image.jpg --mode automatic --output results-automatic
mlx-one track facebook/sam3 frames/ --text person --output results-offline
mlx-one track facebook/sam3 video.mp4 --streaming --prompts-json video-prompts.json --output results-stream
```

Both commands accept `--revision`, `--offline`, and `--cache-dir`. Image prompt JSON
is an object containing `text`, `boxes`, `box_labels`, `points`, `point_labels`, and/or
`mask_path`, as applicable to the mode. Video JSON is an array of records with
`frame_index`, `action` (`add`, `update`, `remove`), text/prompt IDs or object-specific
coordinates/labels/`mask_path`. Unknown fields are rejected. PNG masks and a JSON
manifest record scores, identifiers, boxes, dimensions, and provenance. Existing
output directories are never overwritten; interrupted exports stay marked partial.

If loading fails, verify gated access, the pinned revision, all processor/tokenizer
assets, and float32-compatible tensor names/shapes. If Metal fails in a sandbox,
run natively on the inspected host; a backend-free config test is not GPU evidence.
Variable frame sizes and out-of-bounds coordinates are errors, not implicit rescaling.

## Tests and references

- [Configuration, registry, and strict loading tests](../../../../../tests/test_sam3.py).
- [Inputs, exports, session isolation, correction, cancellation, and cleanup tests](../../../../../tests/test_sam3_workflows.py).
- [Capture runner](../../../../../benchmarks/sam3_qualification.py) and
  [fixed-ID video comparison](../../../../../benchmarks/sam3_compare.py).
- [Publisher source](https://github.com/facebookresearch/sam3),
  [Transformers image reference](https://huggingface.co/docs/transformers/main/en/model_doc/sam3),
  [video reference](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video),
  and [pinned source provenance](NOTICE.md).
