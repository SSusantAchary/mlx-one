"""Compare captured SAM3 video outputs without importing a GPU backend.

Object matching uses one fixed reference-to-native ID mapping, never per-frame
rematching. Passing this check qualifies only the supplied fixture and outputs.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def compare_video(native, reference, *, frames=32, id_mapping=None):
    expected = set(range(frames))

    def indices(data):
        return {
            int(match.group(1))
            for key in data
            if (match := re.fullmatch(r"frame_(\d+)_masks", key))
        }

    native_indices, reference_indices = indices(native), indices(reference)
    mapping = {} if id_mapping is None else dict(id_mapping)
    rows = []
    for index in sorted(native_indices & reference_indices & expected):
        prefix = f"frame_{index}_"
        native_ids = native[prefix + "object_ids"].tolist()
        reference_ids = reference[prefix + "object_ids"].tolist()
        mapped_ids = [mapping.get(identifier, identifier) for identifier in reference_ids]
        matched = len(set(native_ids)) == len(native_ids) and len(set(mapped_ids)) == len(
            mapped_ids
        )
        matched &= set(native_ids) == set(mapped_ids)
        row = dict(frame_index=index, native_ids=native_ids, reference_ids=reference_ids)
        row["detections_match"] = bool(matched)
        metrics = []
        if matched:
            for reference_index, mapped_id in enumerate(mapped_ids):
                native_index = native_ids.index(mapped_id)
                a = native[prefix + "masks"][native_index].astype(bool)
                b = reference[prefix + "masks"][reference_index].astype(bool)
                if a.shape != b.shape:
                    row["detections_match"] = False
                    break
                union = np.logical_or(a, b).sum()
                iou = float(np.logical_and(a, b).sum() / union) if union else 1.0
                score_delta = float(
                    abs(
                        native[prefix + "scores"][native_index]
                        - reference[prefix + "scores"][reference_index]
                    )
                )
                box_delta = float(
                    np.max(
                        np.abs(
                            native[prefix + "boxes"][native_index]
                            - reference[prefix + "boxes"][reference_index]
                        )
                    )
                )
                metrics.append(
                    dict(
                        reference_id=reference_ids[reference_index],
                        native_id=mapped_id,
                        mask_iou=iou,
                        score_delta=score_delta,
                        max_box_delta_px=box_delta,
                        passed=bool(
                            np.isfinite([iou, score_delta, box_delta]).all()
                            and iou >= 0.98
                            and score_delta <= 0.02
                            and box_delta <= 2
                        ),
                    )
                )
        row["objects"] = metrics
        row["passed"] = row["detections_match"] and all(item["passed"] for item in metrics)
        rows.append(row)
    complete = native_indices == expected and reference_indices == expected
    objects = [obj for row in rows for obj in row["objects"]]
    return dict(
        scope="supplied video fixture only; not full SAM3 qualification",
        expected_frames=frames,
        native_frames=sorted(native_indices),
        reference_frames=sorted(reference_indices),
        complete=complete,
        fixed_id_mapping=mapping or "identity",
        gates=dict(mask_iou_min=0.98, score_delta_max=0.02, box_delta_px_max=2),
        passed=complete and all(row["passed"] for row in rows),
        min_mask_iou=min((obj["mask_iou"] for obj in objects), default=None),
        max_score_delta=max((obj["score_delta"] for obj in objects), default=None),
        max_box_delta_px=max((obj["max_box_delta_px"] for obj in objects), default=None),
        frames=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--id-mapping", help='Fixed reference-to-native mapping, e.g. {"0": 0}')
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwrite")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    mapping = None
    if args.id_mapping:
        mapping = {int(key): int(value) for key, value in json.loads(args.id_mapping).items()}
        if len(set(mapping.values())) != len(mapping):
            parser.error("ID mapping must be one-to-one")
    with (
        np.load(args.native, allow_pickle=False) as native,
        np.load(args.reference, allow_pickle=False) as reference,
    ):
        report = compare_video(native, reference, frames=args.frames, id_mapping=mapping)
    report["artifacts"] = {
        name: dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in (("native", args.native), ("reference", args.reference))
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as destination:
            destination.write(rendered)
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
