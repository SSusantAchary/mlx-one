"""Verify the final installed wheel's exact payload and independent tiny outputs."""

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

FORBIDDEN = {
    "torch",
    "transformers",
    "sentence_transformers",
    "FlagEmbedding",
    "kokoro",
    "mlx_lm",
    "mlx_vlm",
}


class NativeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError("unexpected reference-runtime import: " + fullname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, NativeOnly())
    for name in FORBIDDEN:
        try:
            importlib.metadata.distribution(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            continue
        raise AssertionError("reference distribution installed: " + name)
    import mlx_one

    package = Path(mlx_one.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix))
    verified = 0
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if not name.startswith("mlx_one/") or name.endswith("/"):
                continue
            actual = package / name.removeprefix("mlx_one/")
            assert actual.read_bytes() == archive.read(name), name
            verified += 1
    from mlx_one.core.registry import registered_model_types
    from mlx_one.core.release_support import checkpoint_release_status
    from mlx_one.engine.adapter_registry import checkpoint_task_profiles

    expected_types = json.loads(args.inventory.read_text())
    assert list(registered_model_types()) == expected_types
    catalog = json.loads((package / "data/batch_1_sub_6b.json").read_text())
    assert len(catalog["models"]) == 10
    for row in catalog["models"]:
        status = checkpoint_release_status(
            row["model_id"], row["revision"], architecture=row["model_type"]
        )
        assert status["status"] == "SUPPORTED" and status["release_approval"] == "NOT_REQUESTED"
        matches = [
            p
            for p in checkpoint_task_profiles(row["model_type"])
            if p["model_id"] == row["model_id"]
            and p["revision"] == row["revision"]
            and p["task"] == row["task"]
        ]
        assert len(matches) == 1 and matches[0]["functional_tested"]
    import mlx.core as mx
    from mlx.utils import tree_flatten

    assert mx.metal.is_available()
    tiny = {}
    for name, package_name, class_name, config_name in [
        ("t5", "language.t5", "T5ForConditionalGeneration", "T5Config"),
        ("vit", "classification", "ViTForImageClassification", "ViTConfig"),
        ("xlm-embedding", "embeddings.xlm_roberta", "XLMRobertaForRetrieval", "XLMRobertaConfig"),
        ("xlm-reranking", "embeddings.xlm_roberta", "XLMRobertaForRetrieval", "XLMRobertaConfig"),
    ]:
        case = json.loads((args.fixtures / (name + ".json")).read_text())
        config_data = dict(case["config"])
        if name.startswith("xlm"):
            config_data.update(
                task="reranking" if name.endswith("reranking") else "embedding",
                sentence_max_length=8,
            )
        prefix = "mlx_one.models." + package_name
        config_module = importlib.import_module(
            prefix + (".vit_config" if name == "vit" else ".config")
        )
        model_module = importlib.import_module(prefix + (".vit" if name == "vit" else ".model"))
        model = getattr(model_module, class_name)(
            getattr(config_module, config_name).from_dict(config_data)
        )
        weights = {}
        for key, tensor in tree_flatten(model.parameters()):
            phase = int.from_bytes(hashlib.sha256(key.encode()).digest()[:2], "little") / 65536
            value = np.sin(np.arange(tensor.size, dtype=np.float64) * 0.13 + phase) * 0.03
            if "norm" in key.lower() and key.endswith("weight") and tensor.ndim == 1:
                value += 1
            if key == "vit.embeddings.patch_embeddings.projection.weight":
                shape = (tensor.shape[0], tensor.shape[3], tensor.shape[1], tensor.shape[2])
                value = value.reshape(shape).transpose(0, 2, 3, 1)
            else:
                value = value.reshape(tensor.shape)
            weights[key] = mx.array(value.astype(np.float32))
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        if name == "vit":
            output = model(mx.array(case["pixel_values"]))
        else:
            options = dict(attention_mask=mx.array(case["attention_mask"]))
            if name == "t5":
                options["decoder_input_ids"] = mx.array(case["decoder_input_ids"])
            output = model(mx.array(case["input_ids"]), **options)
        actual = np.array(output.logits if "logits" in case else output.last_hidden_state)
        expected = np.asarray(case.get("logits", case.get("hidden")))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        tiny[name] = dict(status="PASS", max_abs_error=float(np.max(np.abs(actual - expected))))
    checked = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert not FORBIDDEN & set(sys.modules)
    result = dict(
        status="PASS",
        wheel=args.wheel.name,
        wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        installed_payload_files=verified,
        package_location="inside isolated environment",
        registered_types=len(expected_types),
        admitted_checkpoint_profiles=10,
        native_tiny_checks=tiny,
        forbidden_reference_distributions="ABSENT",
        pip_check="PASS",
        version=mlx_one.__version__,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
