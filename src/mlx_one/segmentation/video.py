"""Stateful SAM3 video sessions and bounded local frame decoding."""

from __future__ import annotations

import json
import resource
import subprocess
import tempfile
import threading
from functools import wraps
from pathlib import Path

import numpy as np

from .processing import mask_boxes, read_image, validate_coordinates, validate_labels
from .schemas import VideoFrameResult


def iter_frames(source):
    """Yield RGB frames; own and reliably close FFmpeg only for video files."""
    if not isinstance(source, (str, Path)):
        yield from source
        return
    source = Path(source)
    if source.is_dir():
        paths = sorted(
            p
            for p in source.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )
        if not paths:
            raise ValueError("frame directory contains no supported images")
        for path in paths:
            yield read_image(path)
        return
    if not source.is_file():
        raise ValueError(f"video input does not exist: {source}")
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except FileNotFoundError as exc:
        raise RuntimeError("encoded video requires ffmpeg and ffprobe on PATH") from exc
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as exc:
        raise ValueError("cannot inspect the video stream") from exc
    frame_bytes = height * width * 3
    if min(height, width) <= 0 or frame_bytes > 256 * 1024 * 1024:
        raise ValueError("video frame dimensions exceed the decode limit")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            while True:
                data = process.stdout.read(frame_bytes)
                if not data:
                    break
                if len(data) != frame_bytes:
                    raise ValueError("truncated or changing-size decoded frame")
                yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if process.wait() != 0:
                errors.seek(0)
                detail = errors.read(4096).decode(errors="replace")
                raise RuntimeError(f"ffmpeg failed: {detail}")
        finally:
            if process.poll() is None:
                process.terminate()
            if process.stdout:
                process.stdout.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class _DiskFrame:
    def __init__(self, path, processor):
        self.path = path
        self.processor = processor

    def to(self, *args, **kwargs):
        return self.processor.image(self.path)[0].squeeze(0)


def _locked(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        with self._lock:
            return function(self, *args, **kwargs)

    return call


class Sam3VideoSession:
    """One request-owned tracking state; weights may be shared across sessions.

    Offline frames are retained on disk. Streaming has no retroactive correction
    or replay and retains only its latest decoded frame. Model memory selection
    and reference association thresholds are never shortened silently.
    """

    def __init__(self, bundle, frames=None, *, streaming=False, memory_budget_bytes=24 * 1024**3):
        from mlx_one.models.segmentation.sam3._runtime import ops
        from mlx_one.models.segmentation.sam3.video import Sam3VideoInferenceSession

        if bundle.task != "video":
            raise ValueError("video sessions require a video bundle")
        if memory_budget_bytes <= 0:
            raise ValueError("memory budget must be positive")
        self.bundle = bundle
        self._lock = threading.RLock()
        self.streaming = streaming
        self.memory_budget_bytes = memory_budget_bytes
        self.closed = False
        self.cancelled = False
        self.next_frame_index = 0
        self.original_size = None
        self._temp = tempfile.TemporaryDirectory(prefix="mlx-one-sam3-")
        self._paths = {}

        class StreamingState(Sam3VideoInferenceSession):
            seen_frames = 0

            @property
            def num_frames(self):
                return self.seen_frames

        state_class = StreamingState if streaming else Sam3VideoInferenceSession
        self._state = state_class(
            inference_device="mlx",
            inference_state_device="cpu",
            video_storage_device="cpu",
            dtype=ops.float32,
            max_vision_features_cache_size=1,
        )
        self._state.obj_with_new_inputs = []
        try:
            if frames is not None:
                if streaming:
                    raise ValueError(
                        "streaming sessions accept process_frame calls, not preloaded frames"
                    )
                iterator = iter_frames(frames)
                try:
                    for index, frame in enumerate(iterator):
                        self._retain_frame(index, frame)
                finally:
                    if hasattr(iterator, "close"):
                        iterator.close()
                if not self._paths:
                    raise ValueError("video contains no frames")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()

    def _open(self):
        if self.closed:
            raise RuntimeError("video session is closed")
        if self.cancelled:
            raise RuntimeError("video session was cancelled")

    def _dimensions(self, image):
        size = image.height, image.width
        if self.original_size is not None and size != self.original_size:
            raise ValueError("video frame dimensions changed; coordinates cannot be reused")
        self.original_size = size
        self._state.video_height, self._state.video_width = size

    def _retain_frame(self, index, frame):
        image = read_image(frame)
        self._dimensions(image)
        path = Path(self._temp.name) / f"{index:09d}.png"
        image.save(path)
        self._paths[index] = path
        if self._state.processed_frames is None:
            self._state.processed_frames = {}
        self._state.processed_frames[index] = _DiskFrame(path, self.bundle.processor)

    @_locked
    def add_prompt(self, text):
        self._open()
        ids, attention = self.bundle.processor.text(text)
        prompt_id = self._state.add_prompt(text)
        self._state.prompt_input_ids[prompt_id] = ids
        self._state.prompt_attention_masks[prompt_id] = attention
        return prompt_id

    @_locked
    def remove_prompt(self, prompt_id):
        self._open()
        if prompt_id not in self._state.prompts:
            raise ValueError("unknown prompt ID")
        for obj, prompt in list(self._state.obj_id_to_prompt_id.items()):
            if prompt == prompt_id:
                self.remove_object(obj)
        for name in ("prompts", "prompt_input_ids", "prompt_attention_masks", "prompt_embeddings"):
            getattr(self._state, name).pop(prompt_id, None)

    @_locked
    def remove_object(self, object_id):
        self._open()
        self._state.remove_object(object_id, strict=True)
        self._state.obj_id_to_prompt_id.pop(object_id, None)

    @_locked
    def correct_object(
        self,
        object_id,
        frame_index,
        *,
        points=None,
        point_labels=None,
        box=None,
        mask=None,
        prompt_id=None,
    ):
        from mlx_one.models.segmentation.sam3._runtime import Tensor

        self._open()
        if not isinstance(object_id, int) or object_id < 0:
            raise ValueError("object_id must be a nonnegative integer")
        if self.streaming and frame_index != self.next_frame_index:
            raise ValueError("streaming cannot correct a past frame")
        if not self.streaming and frame_index not in self._paths:
            raise ValueError("correction frame is not retained")
        if self.original_size is None:
            raise ValueError("provide a frame before spatial corrections")
        if mask is not None and (points is not None or box is not None):
            raise ValueError("mask correction cannot be combined with points/box")
        if points is None and box is None and mask is None:
            raise ValueError("correction requires points, box, or mask")
        h, w = self.original_size
        scale = self.bundle.processor.image_size
        coords = []
        labels = []
        if box is not None:
            value = validate_coordinates([box], w, h, boxes=True)[0].reshape(2, 2)
            coords.extend(value)
            labels.extend([2, 3])
        if points is not None:
            value = validate_coordinates(points, w, h)
            lab = validate_labels(point_labels, len(value))
            coords.extend(value)
            labels.extend(lab)
        if prompt_id is not None and prompt_id not in self._state.prompts:
            raise ValueError("unknown prompt ID")
        if prompt_id is None:
            prompt_id = self._state.obj_id_to_prompt_id.get(object_id)
            if prompt_id is None:
                raise ValueError("a new spatially prompted object requires a concept prompt_id")
        obj_idx = self._state.obj_id_to_idx(object_id)
        if prompt_id is not None:
            self._state.obj_id_to_prompt_id[object_id] = prompt_id
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != (h, w) or mask.dtype != bool:
                raise ValueError("video correction mask must be original-resolution boolean pixels")
            self._state.add_mask_inputs(obj_idx, frame_index, Tensor(mask[None, None]))
            self._state.point_inputs_per_obj[obj_idx].pop(frame_index, None)
        else:
            normalized = np.asarray(coords, dtype=np.float32) * np.asarray([scale / w, scale / h])
            self._state.point_inputs_per_obj[obj_idx][frame_index] = {
                "point_coords": Tensor(normalized[None, None].astype(np.float32)),
                "point_labels": Tensor(np.asarray(labels, dtype=np.int64)[None, None]),
            }
            self._state.mask_inputs_per_obj[obj_idx].pop(frame_index, None)
        self._state.obj_with_new_inputs.append(object_id)
        self._state.max_obj_id = max(self._state.max_obj_id, object_id)
        self._state.obj_id_to_score.setdefault(object_id, 1.0)
        if not self.streaming:
            for output in self._state.output_dict_per_obj.values():
                for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    for index in list(output[key]):
                        if index >= frame_index:
                            output[key].pop(index)
            self._state.cache.clear_all()

    def _check_budget(self):
        import mlx.core as mx

        if (
            max(mx.get_peak_memory(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            > self.memory_budget_bytes
        ):
            self.cancel()
            raise MemoryError(
                "SAM3 memory budget exceeded; objects/model memory were not silently dropped"
            )

    def _result(self, output):
        self._open()
        from mlx_one.models.segmentation.sam3._runtime import interpolate, ops

        ids = sorted(output.obj_id_to_mask)
        hidden = (
            set(output.suppressed_obj_ids or ())
            | set(output.removed_obj_ids or ())
            | set(self._state.hotstart_removed_obj_ids)
        )
        ids = [i for i in ids if i not in hidden]
        h, w = self.original_size
        masks = (
            np.asarray(
                interpolate(
                    ops.cat([output.obj_id_to_mask[i] for i in ids]).unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            )
            > 0
            if ids
            else np.zeros((0, h, w), dtype=bool)
        )
        keep = np.flatnonzero(masks.any(axis=(1, 2)))
        masks = masks[keep]
        ids = [ids[i] for i in keep]
        # Reference postprocessing resolves overlaps independently per concept.
        groups = {self._state.obj_id_to_prompt_id.get(i, -1) for i in ids}
        for prompt in groups:
            indices = [
                j for j, i in enumerate(ids) if self._state.obj_id_to_prompt_id.get(i, -1) == prompt
            ]
            if len(indices) < 2:
                continue
            quality = np.asarray([output.obj_id_to_tracker_score.get(ids[j], 0) for j in indices])
            pixel_scores = np.where(masks[indices], quality[:, None, None], 0.0)
            winners = pixel_scores.argmax(axis=0)
            for local, index in enumerate(indices):
                masks[index] &= (winners == local) & (pixel_scores[local] > 0)
        scores = np.asarray([output.obj_id_to_score.get(i, 0) for i in ids], dtype=np.float32)
        self._check_budget()
        return VideoFrameResult(
            masks,
            scores,
            mask_boxes(masks),
            (h, w),
            self.bundle.provenance,
            output.frame_idx,
            tuple(ids),
            tuple(self._state.obj_id_to_prompt_id.get(i, -1) for i in ids),
        )

    def propagate(self, *, start_frame_index=0, before_frame=None):
        self._open()
        if self.streaming:
            raise ValueError("use process_frame in streaming mode")
        if not self._state.prompts and not self._state.obj_ids:
            raise ValueError("add a concept or object prompt before tracking")
        if start_frame_index not in self._paths:
            raise ValueError("start frame is not retained")
        pending = []
        for index in range(start_frame_index, len(self._paths)):
            self._open()
            if before_frame is not None:
                before_frame(self, index)
            output = self._offline_frame(index)
            pending.append(output)
            self._state.hotstart_removed_obj_ids.update(output.removed_obj_ids or ())
            if len(pending) >= max(1, self.bundle.model.hotstart_delay):
                yield self._result(pending.pop(0))
        for output in pending:
            self._open()
            yield self._result(output)

    @_locked
    def _offline_frame(self, index):
        self._open()
        try:
            return self.bundle.model(self._state, frame_idx=index)
        except BaseException:
            self.close()
            raise

    @_locked
    def process_frame(self, frame, *, frame_index=None, before_frame=None):
        self._open()
        if not self.streaming:
            raise ValueError("use propagate for offline sessions")
        if frame_index is not None and frame_index != self.next_frame_index:
            raise ValueError("streaming frame indices must be consecutive")
        pixels, image = self.bundle.processor.image(frame)
        self._dimensions(image)
        self._state.seen_frames = self.next_frame_index + 1
        if before_frame is not None:
            before_frame(self, self.next_frame_index)
        if not self._state.prompts and not self._state.obj_ids:
            raise ValueError("add a concept or object prompt before tracking")
        try:
            output = self.bundle.model(self._state, frame=pixels, frame_idx=self.next_frame_index)
            result = self._result(output)
        except BaseException:
            self.close()
            raise
        self.next_frame_index += 1
        self._state.processed_frames.clear()
        return result

    @_locked
    def reset(self):
        self._open()
        self._state.reset_state()
        self._state.point_inputs_per_obj.clear()
        self._state.mask_inputs_per_obj.clear()
        self._state.obj_with_new_inputs = []
        self._state.hotstart_removed_obj_ids.clear()
        self.next_frame_index = 0
        if self.streaming:
            self._state.seen_frames = 0

    def cancel(self):
        if not self.closed:
            self.cancelled = True
            self.close()

    @_locked
    def close(self):
        if self.closed:
            return
        self._state.reset_state()
        self._state.point_inputs_per_obj.clear()
        self._state.mask_inputs_per_obj.clear()
        self._state.hotstart_removed_obj_ids.clear()
        self._state.processed_frames = None
        self._paths.clear()
        self._temp.cleanup()
        self.closed = True
