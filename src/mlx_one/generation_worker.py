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
    from mlx_one.text.loading import resolve_text_model_type

    native_types = {"gpt2", "lfm2", "lfm2_moe", "openelm", "qwen2", "qwen2_moe", "qwen3", "qwen3_5"}
    if resolve_text_model_type(request["model_id"], revision=request["revision"]) in native_types:
        if request.get("adapter_path"):
            raise RuntimeError("native text adapter loading is not implemented")
        results = native_generate(
            request["model_id"],
            request["prompts"],
            revision=request["revision"],
            options=TextGenerationOptions(max_tokens=request["max_tokens"]),
        )
        predictions = [result.text for result in results]
        Path(request["result_path"]).write_text(
            json.dumps({"predictions": predictions}), encoding="utf-8"
        )
        return

    try:
        from mlx_lm import generate, load
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "this legacy generation path requires: pip install 'mlx-one[legacy-mlx-lm]'"
        ) from exc

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
