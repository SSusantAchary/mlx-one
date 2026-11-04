"""Bounded Qwen2.5 video processor candidate; numerical gate is unresolved.

Frame selection follows pinned Qwen2-VL video_processing_qwen2_vl.py. Pixel
packing reuses the existing native image processor and is independently tested
against the source fast video resize. No video-QA task API is admitted.
"""

from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def sample_frame_indices(total_frames, source_fps, *, num_frames=8):
    if type(total_frames) is not int or not 2 <= total_frames <= 1200:
        raise ValueError("video requires2–1200 source frames")
    if not math.isfinite(source_fps) or not 0 < source_fps <= 120:
        raise ValueError("invalid source frame rate")
    if type(num_frames) is not int or not 2 <= num_frames <= 16:
        raise ValueError("video sampling requires2–16 requested frames")
    count = round(num_frames / 2) * 2
    if count > total_frames:
        raise ValueError("requested sampled frames exceed source frame count")
    return np.arange(0, total_frames, total_frames / count).astype(np.int64)


def decode_video(source, *, num_frames=8):
    """Reuse existing bounded local FFmpeg decoder; only short CFR clips admitted."""
    from mlx_one.segmentation.video import iter_frames

    path = Path(source).expanduser()
    if not path.is_file() or path.stat().st_size > 25 * 1024**2:
        raise ValueError("video must be an existing local file of at most25 MiB")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    if stream["r_frame_rate"] != stream["avg_frame_rate"]:
        raise ValueError("variable-frame-rate video is not admitted")
    if (
        float(stream["duration"]) > 10.1
        or int(stream["width"]) * int(stream["height"]) > 4_000_000
        or stream.get("tags", {}).get("rotate", "0") != "0"
        or any(s.get("rotation", 0) != 0 for s in stream.get("side_data_list", []))
    ):
        raise ValueError("video exceeds10s/4MP budget or has unsupported rotation")
    total = int(stream["nb_frames"])
    indices = sample_frame_indices(total, fps, num_frames=num_frames)
    frames = []
    selected = set(indices.tolist())
    decoder = iter_frames(path)
    try:
        for i, frame in enumerate(decoder):
            if i in selected:
                frames.append(frame.copy())
            if i == int(indices[-1]):
                break
    finally:
        decoder.close()
    if len(frames) != len(indices):
        raise ValueError("video ended before selected frames")
    return np.stack(frames), {
        "frame_indices": indices.tolist(),
        "timestamps": [int(i) / fps for i in indices],
        "source_fps": fps,
        "source_frames": total,
        "sampled_fps": len(indices) / total * fps,
    }


def pack_video_frames(frames, processor):
    """CTHW temporal pairs from existing native per-frame image patch layout."""
    import mlx.core as mx
    from PIL import Image

    frames = np.asarray(frames)
    if (
        frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or not 1 <= len(frames) <= 16
    ):
        raise ValueError("video frames require1–16 RGB uint8 frames [T,H,W,3]")
    if frames.shape[1] * frames.shape[2] > 4_000_000:
        raise ValueError("video frames exceed4MP")
    values = []
    grid = None
    for frame in frames:
        with Image.fromarray(frame) as image:
            patches, current = processor._patches(image, mx)
        if grid is not None and current != grid:
            raise ValueError("video frames must have identical processed dimensions")
        grid = current
        patch = processor.config.vision_config.patch_size
        values.append(patches.reshape(-1, 3, 2, patch, patch)[:, :, 0])
    if len(values) % 2:
        values.append(values[-1])
    pairs = [
        mx.stack(values[i : i + 2], axis=2).reshape(-1, 3 * 2 * patch * patch)
        for i in range(0, len(values), 2)
    ]
    return mx.concatenate(pairs, axis=0), ((len(values) // 2, grid[1], grid[2]),)
