"""Host mask NMS and eight-connected components; shared with reference validation."""

import numpy as np


def mask_nms(ious, scores, threshold, use_iou_matrix=True):
    matrix = np.asarray(ious)
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept = []
    for i in order:
        if all(matrix[i, j] <= threshold for j in kept):
            kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def connected_components(mask, get_counts=True):
    data = np.asarray(mask).astype(bool)
    labels = np.zeros(data.shape, dtype=np.int32)
    counts = np.zeros_like(labels)
    for batch in range(data.shape[0]):
        for channel in range(data.shape[1]):
            src = data[batch, channel]
            dst = labels[batch, channel]
            h, w = src.shape
            parents = [0]
            sizes = [0]
            runs = []
            previous = []

            def find(x, parents=parents):
                while parents[x] != x:
                    parents[x] = parents[parents[x]]
                    x = parents[x]
                return x

            for y in range(h):
                transitions = np.diff(np.pad(src[y].astype(np.int8), (1, 1)))
                starts = np.flatnonzero(transitions == 1)
                ends = np.flatnonzero(transitions == -1)
                current = []
                for start, end in zip(starts, ends, strict=True):
                    label = len(parents)
                    parents.append(label)
                    sizes.append(int(end - start))
                    for a, b, other in previous:
                        if b >= start and a <= end:
                            root = find(other)
                            own = find(label)
                            if root != own:
                                parents[root] = own
                                sizes[own] += sizes[root]
                    current.append((int(start), int(end), label))
                    runs.append((y, int(start), int(end), label))
                previous = current
            for y, start, end, label in runs:
                root = find(label)
                dst[y, start:end] = root
                counts[batch, channel, y, start:end] = sizes[root]
    return labels, counts
