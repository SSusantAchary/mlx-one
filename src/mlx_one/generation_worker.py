"""Isolated deterministic MLX text-generation worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m mlx_one.generation_worker REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    from mlx_lm import generate, load

    model, tokenizer = load(
        request["model_id"],
        revision=request["revision"],
        adapter_path=request.get("adapter_path"),
    )
    predictions = [
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request["max_tokens"],
            verbose=False,
        )
        for prompt in request["prompts"]
    ]
    Path(request["result_path"]).write_text(
        json.dumps({"predictions": predictions}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
