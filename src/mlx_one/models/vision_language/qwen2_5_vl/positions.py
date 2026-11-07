"""Source-pinned Qwen2.5 video temporal positions, separate from Qwen2-VL.

Transformers 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc get_rope_index.
Apache-2.0. Preserve its seconds-to-integer cast, including fractional intervals.
"""

import math


def video_position_ids(
    input_ids,
    *,
    mm_token_type_ids,
    image_grid_thw,
    video_grid_thw,
    spatial_merge_size,
    attention_mask,
    second_per_grid_ts,
    tokens_per_second,
):
    import mlx.core as mx

    grids = list(video_grid_thw)
    seconds = [1.0] * len(grids) if second_per_grid_ts is None else list(second_per_grid_ts)
    if len(seconds) != len(grids):
        raise ValueError("one seconds-per-grid interval is required per video")
    if any(not math.isfinite(float(s)) or float(s) <= 0 for s in seconds):
        raise ValueError("video grid intervals must be finite and positive")
    images, videos = iter(image_grid_thw or ()), iter(zip(grids, seconds, strict=True))
    batch, width = input_ids.shape
    positions, deltas = [], []
    for b in range(batch):
        types = mm_token_type_ids[b].tolist()
        valid = [True] * width if attention_mask is None else attention_mask[b].tolist()
        row, cursor, index = [[1] * width for _ in range(3)], 0, 0
        while index < width:
            if not valid[index]:
                index += 1
                continue
            kind, stop = int(types[index]), index
            while stop < width and valid[stop] and int(types[stop]) == kind:
                stop += 1
            length = stop - index
            if kind == 0:
                for local in range(length):
                    for axis in range(3):
                        row[axis][index + local] = cursor + local
                cursor += length
            else:
                try:
                    if kind == 1:
                        (temporal, height, width_grid), step = next(images), 0
                    elif kind == 2:
                        (temporal, height, width_grid), second = next(videos)
                        # Pinned source casts interval to range_tensor.dtype (int64)
                        # before multiplying by tokens_per_second. Do not "fix" it.
                        step = int(float(second)) * tokens_per_second
                    else:
                        raise ValueError("unsupported multimodal token type")
                except StopIteration as exc:
                    raise ValueError("missing media grid for token segment") from exc
                temporal, height, width_grid = int(temporal), int(height), int(width_grid)
                h, w = height // spatial_merge_size, width_grid // spatial_merge_size
                if (
                    min(temporal, h, w) < 1
                    or height % spatial_merge_size
                    or width_grid % spatial_merge_size
                ):
                    raise ValueError("invalid multimodal grid")
                if temporal * h * w != length:
                    raise ValueError("media token segment differs from grid")
                local = 0
                for t in range(temporal):
                    for y in range(h):
                        for x in range(w):
                            row[0][index + local], row[1][index + local], row[2][index + local] = (
                                cursor + int(t * step),
                                cursor + y,
                                cursor + x,
                            )
                            local += 1
                cursor += max(int((temporal - 1) * step) + 1, h, w)
            index = stop
        positions.append(mx.array(row, dtype=mx.int64))
        deltas.append(cursor - width)
    return mx.stack(positions, axis=1), mx.array(deltas, dtype=mx.int64)[:, None]
