"""Pinned safetensors discovery and strict native component loading."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import tempfile
from dataclasses import dataclass
from math import prod
from pathlib import Path

from .profiles import ImageProfile, artifact_lock, published_q4_lock

MANIFEST = "mlx-one-image.json"
GROUP_SIZE = 64


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint key: {key}")
        result[key] = value
    return result


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path.name}")
    return value


def file_digest(path: Path) -> str:
    """Compute SHA-256 using bounded file reads."""
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024**2), b""):
            digest.update(chunk)
        return digest.hexdigest()


def tensor_header(path: Path) -> dict:
    """Read tensor names, shapes and dtypes without allocating weight buffers."""
    with path.open("rb") as source:
        prefix = source.read(8)
        size = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else 0
        if not 2 <= size <= 16 * 1024**2:
            raise ValueError(f"invalid safetensors header in {path.name}")
        header = json.loads(source.read(size), object_pairs_hook=_unique_pairs)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors tensor inventory in {path.name}")
    header.pop("__metadata__", None)
    sizes = {"BF16": 2, "F32": 4, "I64": 8, "U32": 4}
    for spec in header.values():
        if (
            not isinstance(spec, dict)
            or spec.keys() != {"dtype", "shape", "data_offsets"}
            or not isinstance(spec["dtype"], str)
            or spec["dtype"] not in sizes
            or not isinstance(spec["shape"], list)
            or any(type(axis) is not int or axis < 0 for axis in spec["shape"])
            or not isinstance(spec["data_offsets"], list)
            or len(spec["data_offsets"]) != 2
            or any(type(offset) is not int for offset in spec["data_offsets"])
        ):
            raise ValueError(f"invalid safetensors tensor metadata in {path.name}")
    end = 0
    for spec in sorted(header.values(), key=lambda value: value["data_offsets"]):
        start, stop = spec["data_offsets"]
        if start != end or stop - start != prod(spec["shape"]) * sizes[spec["dtype"]]:
            raise ValueError(f"invalid safetensors offsets in {path.name}")
        end = stop
    if 8 + size + end != path.stat().st_size:
        raise ValueError(f"incomplete safetensors file: {path.name}")
    return {key: [spec["dtype"], spec["shape"]] for key, spec in header.items()}


def header_digest(header: dict) -> str:
    """Hash the canonical set of checkpoint tensor names, dtypes and shapes."""
    return hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ImageCheckpoint:
    """Resolved checkpoint with verified configuration and declared packed layout."""

    root: Path
    profile: ImageProfile
    lock: dict
    packed_bits: int | None
    converted: dict
    published: bool = False

    def config(self, component: str) -> dict:
        """Return the validated configuration belonging to a pipeline component."""
        if self.published:
            return self.lock["configs"][component]
        filename = "scheduler_config.json" if component == "scheduler" else "config.json"
        return _json(self.root / component / filename)

    def shards(self, component: str) -> dict:
        """Return the pinned shard names and their verification records."""
        return self.lock["weights"][component]


def resolve_checkpoint(
    profile: ImageProfile,
    checkpoint: str | Path | None,
    bits: int | None,
    *,
    offline: bool,
    published: bool = False,
) -> ImageCheckpoint:
    """Resolve a pinned original, local export, or published Q4 checkpoint."""
    if published and (bits != 4 or profile.q4_model is None):
        raise ValueError("the published image checkpoint requires Q4")
    lock = published_q4_lock(profile) if published else artifact_lock(profile)
    model = profile.q4_model if published else profile.model
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        names = list(lock["files"]) + [
            name for shards in lock["weights"].values() for name in shards
        ]
        root = Path(
            snapshot_download(
                model,
                revision=profile.artifact_revision(model),
                allow_patterns=names,
                local_files_only=offline,
            )
        )
    else:
        root = Path(checkpoint).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"image checkpoint directory does not exist: {root}")
    for name, expected in lock["files"].items():
        if not (root / name).is_file():
            raise ValueError(f"image checkpoint is missing a pinned asset: {name}")
        if file_digest(root / name) != expected:
            raise ValueError(f"image checkpoint asset differs from the pinned revision: {name}")
    converted = {}
    manifest = root / MANIFEST
    if manifest.is_file():
        if published:
            raise ValueError("a local MLX-One export must use the original image model ID")
        data = _json(manifest)
        if (
            bits not in (4, 8)
            or type(data.get("bits")) is not int
            or data != _manifest(profile, bits, data.get("shards"))
            or not isinstance(data["shards"], dict)
            or set(data["shards"]) != set(lock["weights"]["transformer"])
        ):
            raise ValueError("image conversion manifest does not match the requested precision")
        converted = data["shards"]
    for component in lock["weights"].values():
        for name in component:
            if not (root / name).is_file():
                raise ValueError(f"image checkpoint is missing a pinned weight shard: {name}")
    return ImageCheckpoint(
        root,
        profile,
        lock,
        bits if converted or published else None,
        converted,
        published=published,
    )


def _manifest(profile, bits, shards):
    return {
        "format": "mlx-one-image-affine-v1",
        "model": profile.model,
        "revision": profile.revision,
        "bits": bits,
        "precision": profile.precision(bits),
        "shards": shards,
    }


def _source_headers(checkpoint: ImageCheckpoint, component: str):
    seen = set()
    for name, expected in checkpoint.shards(component).items():
        path = checkpoint.root / name
        header = tensor_header(path)
        if header_digest(header) != expected:
            raise ValueError(f"unexpected tensors, shapes or dtypes in {name}")
        if seen.intersection(header):
            raise ValueError(f"duplicate tensors across {component} shards")
        seen.update(header)
        yield name, path, header


def _quantized_path(path: str) -> bool:
    return path.startswith(
        (
            "transformer_blocks.",
            "single_transformer_blocks.",
            "layers.",
            "noise_refiner.",
            "context_refiner.",
        )
    ) and any(
        part in path.split(".")
        for part in ("attn", "ff", "ff_context", "attention", "feed_forward")
    )


def _packable(key, shape):
    return (
        key.endswith(".weight")
        and len(shape) == 2
        and shape[1] % GROUP_SIZE == 0
        and _quantized_path(key)
    )


def _weight_iter(checkpoint, component, bits, expected):
    import mlx.core as mx

    if component == "transformer" and checkpoint.packed_bits:
        for name, identity in checkpoint.converted.items():
            path = checkpoint.root / name
            if (
                not isinstance(identity, dict)
                or identity.keys() != {"layout", "sha256"}
                or header_digest(tensor_header(path)) != identity["layout"]
                or file_digest(path) != identity["sha256"]
            ):
                raise ValueError(f"converted shard differs from manifest: {name}")
            yield from mx.load(path).items()
        return
    for _, path, header in _source_headers(checkpoint, component):
        # MLX loads safetensors lazily. Materialize attached weights in load_component,
        # keeping the dense source of a quantized matrix live for that matrix only.
        for source_key, tensor in mx.load(path).items():
            name = source_key.removeprefix("model.") if component == "text_encoder" else source_key
            if name not in expected:
                # The pinned source contract validates intentionally unused text/VAE tensors.
                if component == "transformer":
                    raise ValueError(f"unmapped transformer tensor: {name}")
                continue
            if component == "vae" and tensor.ndim == 4:
                tensor = tensor.transpose(0, 2, 3, 1)
            if component == "transformer":
                tensor = tensor.astype(mx.bfloat16)
            if bits and component == "transformer" and _packable(name, header[source_key][1]):
                packed = mx.quantize(tensor, group_size=GROUP_SIZE, bits=bits, mode="affine")
                mx.eval(*packed)
                for suffix, value in zip(("weight", "scales", "biases"), packed, strict=True):
                    yield f"{name.removesuffix('.weight')}.{suffix}", value
            else:
                yield name, tensor


def load_component(checkpoint: ImageCheckpoint, component: str, module, bits: int | None = None):
    """Match complete native parameters and evaluate each weight before attachment."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    if checkpoint.published:
        from .prepacked import prepare_published_weights

        source = prepare_published_weights(checkpoint, component, module)
    elif component == "transformer" and bits:
        nn.quantize(
            module,
            group_size=GROUP_SIZE,
            bits=bits,
            mode="affine",
            class_predicate=lambda name, layer: (
                isinstance(layer, nn.Linear) and _packable(f"{name}.weight", layer.weight.shape)
            ),
        )
    expected = dict(tree_flatten(module.parameters()))
    seen = set()
    weights = []
    if not checkpoint.published:
        source = _weight_iter(checkpoint, component, bits, expected)
    for name, tensor in source:
        if name in seen or name not in expected:
            raise ValueError(f"duplicate or unexpected {component} tensor: {name}")
        shape = expected[name].shape
        dtype = mx.uint32 if expected[name].dtype == mx.uint32 else mx.bfloat16
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{component} tensor {name} has {tensor.shape}/{tensor.dtype}, "
                f"expected {shape}/{dtype}"
            )
        mx.eval(tensor)
        weights.append((name, tensor))
        seen.add(name)
    if missing := expected.keys() - seen:
        raise ValueError(f"missing {component} tensors: {', '.join(sorted(missing)[:8])}")
    module.load_weights(weights, strict=True)
    module.eval()
    return module


def convert_checkpoint(
    profile: ImageProfile,
    source: str | Path,
    destination: str | Path,
    *,
    bits: int,
) -> Path:
    """Export a self-contained affine checkpoint one transformer shard at a time."""
    from mlx_one.engine.device_executor import DeviceExecutor

    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("conversion supports affine 4-bit and 8-bit weights")
    checkpoint = resolve_checkpoint(profile, source, None, offline=True)
    dest = Path(destination).expanduser().resolve()
    return DeviceExecutor.shared().call(
        ("image-convert", dest), lambda: _export(checkpoint, dest, bits)
    )


def _export(checkpoint: ImageCheckpoint, dest: Path, bits: int) -> Path:
    import mlx.core as mx

    if dest.exists():
        raise FileExistsError(f"conversion destination already exists: {dest}")
    with tempfile.TemporaryDirectory(prefix=".image-convert-", dir=dest.parent) as staging:
        root = Path(staging) / "checkpoint"
        root.mkdir()
        for name in checkpoint.lock["files"]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint.root / name, target)
        for component in ("text_encoder", "vae"):
            for name, path, _ in _source_headers(checkpoint, component):
                shutil.copyfile(path, root / name)
        converted = {}
        for name, path, header in _source_headers(checkpoint, "transformer"):
            packed = {}
            for key, weight in mx.load(path).items():
                weight = weight.astype(mx.bfloat16)
                if _packable(key, header[key][1]):
                    values = mx.quantize(weight, group_size=GROUP_SIZE, bits=bits, mode="affine")
                    mx.eval(*values)
                    for suffix, value in zip(("weight", "scales", "biases"), values, strict=True):
                        packed[f"{key.removesuffix('.weight')}.{suffix}"] = value
                else:
                    mx.eval(weight)
                    packed[key] = weight
            mx.save_safetensors(root / name, packed)
            converted[name] = {
                "layout": header_digest(tensor_header(root / name)),
                "sha256": file_digest(root / name),
            }
            del packed
            mx.clear_cache()
        (root / MANIFEST).write_text(
            json.dumps(_manifest(checkpoint.profile, bits, converted), indent=2)
        )
        root.rename(dest)
    return dest
