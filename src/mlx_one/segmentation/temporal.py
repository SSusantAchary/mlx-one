"""Point/box-prompted standalone SAM3 temporal tracking over the native port."""

from __future__ import annotations

import resource
import tempfile
import threading
from pathlib import Path

import numpy as np

from .processing import mask_boxes, read_image, validate_coordinates, validate_labels
from .schemas import VideoFrameResult
from .video import _DiskFrame, _locked, iter_frames


class Sam3TemporalSession:
    """Own temporal memories, prompt identities and disk-backed input frames."""

    def __init__(
        self,
        bundle,
        frames=None,
        *,
        streaming=False,
        memory_budget_bytes=24 * 1024**3,
        original_size=None,
    ):
        from mlx_one.models.segmentation.sam3._runtime import ops
        from mlx_one.models.segmentation.sam3.tracker_video import Sam3TrackerVideoInferenceSession

        if bundle.task != "tracking" or memory_budget_bytes <= 0:
            raise ValueError(
                "temporal sessions require a tracking bundle and positive memory budget"
            )
        self.bundle, self.streaming = bundle, streaming
        self.memory_budget_bytes = memory_budget_bytes
        self._lock = threading.RLock()
        self.closed = self.cancelled = False
        self.next_frame_index = 0
        self.original_size = original_size
        self._temp = tempfile.TemporaryDirectory(prefix="mlx-one-temporal-")
        self._paths = {}
        self._state = Sam3TrackerVideoInferenceSession(
            inference_device="mlx",
            inference_state_device="cpu",
            video_storage_device="cpu",
            dtype=ops.float32,
            max_vision_features_cache_size=1,
        )
        try:
            if frames is not None:
                if streaming:
                    raise ValueError("streaming temporal sessions accept process_frame calls")
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
        if self.closed or self.cancelled:
            raise RuntimeError("temporal session is closed or cancelled")

    def _retain_frame(self, index, frame):
        image = read_image(frame)
        size = (image.height, image.width)
        if self.original_size is not None and size != tuple(self.original_size):
            raise ValueError("video frame dimensions changed")
        self.original_size = size
        self._state.video_height, self._state.video_width = size
        path = Path(self._temp.name) / f"{index:09d}.png"
        image.save(path)
        self._paths[index] = path
        if self._state.processed_frames is None:
            self._state.processed_frames = {}
        self._state.processed_frames[index] = _DiskFrame(path, self.bundle.processor)

    def add_prompt(self, text):
        raise ValueError("standalone temporal tracking requires point or box object prompts")

    def remove_prompt(self, prompt_id):
        raise ValueError("standalone temporal tracking has object prompts, not text prompts")

    def remove_object(self, object_id):
        raise ValueError(
            "object removal is not admitted in this temporal profile; reset the session"
        )

    @_locked
    def add_object(self, object_id, frame_index=0, *, points=None, point_labels=None, box=None):
        from mlx_one.models.segmentation.sam3._runtime import Tensor

        self._open()
        if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id < 0:
            raise ValueError("object_id must be a nonnegative integer")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if self.original_size is None:
            raise ValueError("first streamed prompt requires original_size or a frame callback")
        if not self.streaming and frame_index not in self._paths:
            raise ValueError("prompt frame is outside the video")
        if self.streaming and frame_index != self.next_frame_index:
            raise ValueError("streamed prompts must address the next frame")
        h, w = self.original_size
        coords, labels = [], []
        if box is not None:
            coords.extend(validate_coordinates([box], w, h, boxes=True)[0].reshape(2, 2))
            labels.extend([2, 3])
        if points is not None:
            values = validate_coordinates(points, w, h)
            coords.extend(values)
            labels.extend(validate_labels(point_labels, len(values)))
        if not coords:
            raise ValueError("temporal tracking requires point or box prompts")
        index = self._state.obj_id_to_idx(object_id)
        size = self.bundle.processor.image_size
        scaled = np.asarray(coords, dtype=np.float32) * np.asarray([size / w, size / h])
        self._state.point_inputs_per_obj[index][frame_index] = dict(
            point_coords=Tensor(scaled.astype(np.float32)[None, None]),
            point_labels=Tensor(np.asarray(labels, dtype=np.int64)[None, None]),
        )
        self._state.mask_inputs_per_obj[index].pop(frame_index, None)
        if object_id not in self._state.obj_with_new_inputs:
            self._state.obj_with_new_inputs.append(object_id)
        for output in self._state.output_dict_per_obj.values():
            for group in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for frame in list(output[group]):
                    if frame >= frame_index:
                        del output[group][frame]
        self._state.cache.clear_all()
        return object_id

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
        if mask is not None or prompt_id is not None:
            raise ValueError("standalone temporal tracking supports point and box prompts")
        return self.add_object(
            object_id, frame_index, points=points, point_labels=point_labels, box=box
        )

    def _result(self, output):
        from mlx_one.models.segmentation.sam3._runtime import interpolate

        values = interpolate(
            output.pred_masks, size=self.original_size, mode="bilinear", align_corners=False
        )
        logits = np.asarray(values)[:, 0]
        masks = logits > 0
        scores = (1 / (1 + np.exp(-np.asarray(output.object_score_logits)))).reshape(-1)
        return VideoFrameResult(
            masks,
            scores,
            mask_boxes(masks),
            self.original_size,
            self.bundle.provenance,
            int(output.frame_idx),
            tuple(output.object_ids),
            (),
        )

    def _budget(self):
        import mlx.core as mx

        if (
            max(mx.get_peak_memory(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            > self.memory_budget_bytes
        ):
            self.cancelled = True
            raise MemoryError("temporal tracking memory budget exceeded")

    @_locked
    def process_frame(self, frame, *, frame_index=None, before_frame=None):
        self._open()
        if not self.streaming:
            raise ValueError("process_frame requires a streaming session")
        index = self.next_frame_index if frame_index is None else frame_index
        if index != self.next_frame_index:
            raise ValueError("streamed frames must be consecutive")
        self._retain_frame(index, frame)
        if before_frame is not None:
            before_frame(self, index)
        if not self._state.obj_ids:
            raise ValueError("add objects before tracking")
        pixels = self._state.get_frame(index)
        output = self.bundle.model(self._state, frame_idx=index, frame=pixels)
        result = self._result(output)
        self._state.processed_frames[index] = _DiskFrame(self._paths[index], self.bundle.processor)
        self.next_frame_index += 1
        self._budget()
        return result

    def propagate(self, *, start_frame_index=0, before_frame=None):
        self._open()
        if self.streaming or not self._state.obj_ids:
            raise ValueError("offline propagation requires frames and object prompts")
        for index in range(start_frame_index, len(self._paths)):
            with self._lock:
                self._open()
                if before_frame is not None:
                    before_frame(self, index)
                result = self._result(self.bundle.model(self._state, frame_idx=index))
                self._budget()
            yield result

    @_locked
    def reset(self):
        self._open()
        self._state.reset_inference_session()
        self.next_frame_index = 0

    def cancel(self):
        self.cancelled = True

    @_locked
    def close(self):
        if not self.closed:
            self._state.reset_inference_session()
            self._state.processed_frames = None
            self._paths.clear()
            self._temp.cleanup()
            self.closed = True
