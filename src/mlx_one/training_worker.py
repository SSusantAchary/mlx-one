"""Isolated bridge to the pinned-revision mlx-lm LoRA trainer."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m mlx_one.training_worker REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    model_ref = request["model_id"]
    if not Path(model_ref).exists():
        from huggingface_hub import snapshot_download

        model_ref = snapshot_download(model_ref, revision=request["revision"])

    try:
        from mlx_lm.lora import CONFIG_DEFAULTS, run
    except ModuleNotFoundError as exc:
        from mlx_one.compat.dependencies import legacy_mlx_lm_error

        raise legacy_mlx_lm_error("Legacy LoRA training") from exc

    values = dict(CONFIG_DEFAULTS)
    values.update(request["mlx_lm_config"])
    values["model"] = model_ref
    run(types.SimpleNamespace(**values))


if __name__ == "__main__":
    main()
