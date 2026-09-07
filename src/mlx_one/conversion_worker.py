"""Isolated bridge to the revision-aware mlx-lm conversion API."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Convert one pinned Hugging Face checkpoint to a local MLX artifact."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m mlx_one.conversion_worker REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from huggingface_hub import snapshot_download
    from mlx_lm.convert import convert

    snapshot = snapshot_download(request["model_id"], revision=request["revision"])
    convert(
        hf_path=snapshot,
        mlx_path=request["destination"],
        dtype="bfloat16",
        quantize=bool(request["quantize"]),
        q_bits=4 if request["quantize"] else None,
        q_group_size=64 if request["quantize"] else None,
        trust_remote_code=False,
    )


if __name__ == "__main__":
    main()
