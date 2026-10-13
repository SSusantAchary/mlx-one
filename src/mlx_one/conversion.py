"""Safe native conversion of registered causal-language checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


class ConversionError(RuntimeError):
    """Raised when a checkpoint cannot be converted safely."""


def convert_text_model(
    source: str | Path,
    destination: str | Path,
    *,
    revision: str | None = None,
    quantize: bool = False,
    group_size: int = 64,
    bits: int = 4,
    max_shard_size: int = 5 * 1024**3,
) -> Path:
    """Convert safetensors to a native BF16 or affine INT4/INT8 MLX bundle."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ConversionError("group_size must be a positive integer")
    if (
        isinstance(max_shard_size, bool)
        or not isinstance(max_shard_size, int)
        or max_shard_size < 1
    ):
        raise ConversionError("max_shard_size must be a positive integer")
    if bits not in {4, 8} or isinstance(bits, bool) or bits == 8 and not quantize:
        raise ConversionError("bits=8 requires quantize=True; supported bits are 4 and 8")
    target = Path(destination)
    if target.exists():
        raise ConversionError(f"conversion destination already exists: {target}")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        from mlx_one.text import load_text_model

        bundle = load_text_model(source, revision=revision)
        if quantize:
            if bits == 8:
                bundle.model.set_dtype(mx.float16)
            nn.quantize(bundle.model, group_size=group_size, bits=bits, mode="affine")
        parameters = dict(tree_flatten(bundle.model.parameters()))
        if not quantize:
            floating = {mx.float16, mx.float32, mx.bfloat16}
            parameters = {
                name: value.astype(mx.bfloat16) if value.dtype in floating else value
                for name, value in parameters.items()
            }
        staging = target.parent / f".{target.name}.conversion"
        if staging.exists():
            raise ConversionError(f"stale conversion staging directory exists: {staging}")
        staging.mkdir(parents=True)
        try:
            for path in bundle.path.iterdir():
                if path.is_file() and path.name.endswith(
                    (".json", ".jinja", ".txt", ".model")
                ):
                    shutil.copy2(path, staging / path.name)
            config_path = staging / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if quantize:
                config["quantization"] = {
                    "bits": bits,
                    "group_size": group_size,
                    "mode": "affine",
                    "quant_method": "mlx",
                }
            else:
                config.pop("quantization", None)
                config.pop("quantization_config", None)
                config["torch_dtype"] = "bfloat16"
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _save_shards(mx, staging, parameters, max_shard_size)
            policy = {"schema_version": 2, "source": getattr(bundle, "model_id", str(source)),
                "revision": getattr(bundle, "revision", revision),
                "precision": f"int{bits}-affine" if quantize else "bfloat16",
                "group_size": group_size if quantize else None,
                "converter": "mlx-one-native-v2",
                "floating_dtype": "float16" if bits == 8 else None}
            write_conversion_manifest(staging, policy)
            staging.replace(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"native conversion failed: {exc}") from exc
    return target


def _save_shards(
    mx: Any, root: Path, parameters: dict[str, Any], maximum: int
) -> None:
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_size = 0
    for name in sorted(parameters):
        value = parameters[name]
        size = int(value.nbytes)
        if current and current_size + size > maximum:
            groups.append(current)
            current, current_size = {}, 0
        current[name] = value
        current_size += size
    if current:
        groups.append(current)
    weight_map = {}
    total = len(groups)
    for index, tensors in enumerate(groups, start=1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        mx.save_safetensors(str(root / filename), tensors)
        weight_map.update({name: filename for name in tensors})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(int(value.nbytes) for value in parameters.values())
                },
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_conversion_manifest(root, policy):
    """Bind saved weights and tokenizer/config assets to the conversion policy."""
    root = Path(root)
    manifest = {**policy, "files": {p.name: _file_digest(p) for p in sorted(root.iterdir())
        if p.is_file() and p.name != "mlx-one-conversion.json"}}
    manifest["artifact_digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    (root / "mlx-one-conversion.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest


def verify_conversion_manifest(root):
    root = Path(root)
    path = root / "mlx-one-conversion.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        return None  # historical conversion records remain diagnostic
    payload = {k: v for k, v in manifest.items() if k != "artifact_digest"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if expected != manifest.get("artifact_digest"):
        raise ConversionError("conversion identity checksum mismatch")
    files = manifest.get("files", {})
    actual = {p.name for p in root.iterdir() if p.is_file()
              and p.name != "mlx-one-conversion.json"}
    if not files or set(files) != actual:
        raise ConversionError("conversion asset inventory mismatch")
    for name, checksum in files.items():
        if Path(name).name != name or _file_digest(root / name) != checksum:
            raise ConversionError("conversion asset checksum mismatch")
    return manifest
