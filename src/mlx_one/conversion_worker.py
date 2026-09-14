"""Isolated worker for native mlx-one checkpoint conversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Convert one pinned Hugging Face checkpoint to a local MLX artifact."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m mlx_one.conversion_worker REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from mlx_one.conversion import convert_text_model

    convert_text_model(
        request["model_id"],
        request["destination"],
        revision=request["revision"],
        quantize=bool(request["quantize"]),
        group_size=64,
    )


if __name__ == "__main__":
    main()
