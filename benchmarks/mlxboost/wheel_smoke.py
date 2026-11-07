"""Run with the clean wheel interpreter, without adding the repository to sys.path."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx_one
    from mlx_one import Engine
    from mlx_one.mlxboost import MLXBoostConfig
    from mlx_one.mlxboost.provenance import source_manifest
    from mlx_one.registry import InferenceFeatureRegistry

    with Engine(
        args.model,
        offline=True,
        inference_mode="mlxboost_b1",
        execution_dtype="float16",
        qualification_registry=InferenceFeatureRegistry(),
        mlxboost_config=MLXBoostConfig(fast_decode=True),
    ) as engine:
        result = engine.generate("A research team studied", max_tokens=8, temperature=0)
        selection = engine.capabilities()["selection"]
    installed = {
        d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()
    }
    forbidden = {"mlx-lm", "omlx", "transformers"}
    assert not installed & forbidden
    assert not {"mlx_lm", "omlx", "transformers"} & set(sys.modules)
    assert "site-packages" in mlx_one.__file__
    assert len(source_manifest()["upstream_repositories"]) == 4
    assert result.generation_tokens == 8
    assert selection["resolved_mode"] == "mlxboost_b1"
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS",
                "package_file": mlx_one.__file__,
                "text": result.text,
                "tokens": result.generation_tokens,
                "selection": selection,
                "installed": sorted(installed),
                "forbidden_imports": [],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
