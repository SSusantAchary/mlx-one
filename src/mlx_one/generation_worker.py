"""Isolated deterministic MLX text-generation worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m mlx_one.generation_worker REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    from mlx_one.text import TextGenerationOptions
    from mlx_one.text import generate as native_generate

    results = native_generate(
        request["model_id"],
        request["prompts"],
        options=TextGenerationOptions(max_tokens=request["max_tokens"]),
        revision=request["revision"],
        adapter_path=request.get("adapter_path"),
    )
    predictions = [result.text for result in results]
    Path(request["result_path"]).write_text(
        json.dumps({"predictions": predictions}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
