"""Non-overwriting mask export with explicit partial/complete manifests."""

import json
from pathlib import Path

import numpy as np


class ResultWriter:
    def __init__(self, output):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.records = []
        self.status = "partial"
        self._write()

    def _write(self):
        (self.root / "manifest.json").write_text(
            json.dumps({"status": self.status, "results": self.records}, indent=2) + "\n"
        )

    def add(self, result):
        from PIL import Image

        frame = getattr(result, "frame_index", None)
        prefix = f"frame-{frame:09d}" if frame is not None else "image"
        paths = []
        for index, mask in enumerate(result.masks):
            name = f"{prefix}-mask-{index:05d}.png"
            if (self.root / name).exists():
                raise FileExistsError(f"mask output already exists: {name}")
            Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(self.root / name)
            paths.append(name)
        record = {
            "masks": paths,
            "boxes": np.asarray(result.boxes).tolist(),
            "original_size": list(result.original_size),
            "provenance": result.provenance,
        }
        if hasattr(result, "predicted_iou"):
            record["predicted_iou"] = np.asarray(result.predicted_iou).tolist()
        else:
            record["scores"] = np.asarray(result.scores).tolist()
        if frame is not None:
            record.update(
                frame_index=frame,
                object_ids=list(result.object_ids),
                prompt_ids=list(result.prompt_ids),
            )
        self.records.append(record)
        self._write()

    def complete(self):
        self.status = "complete"
        self._write()
