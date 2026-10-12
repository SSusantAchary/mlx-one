"""Backend-free selection and validation of safe audio checkpoint layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AudioWeightsError(ValueError):
    """An audio checkpoint has an incomplete or ambiguous tensor layout."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AudioWeightsError(f"duplicate checkpoint index key: {key}")
        result[key] = value
    return result


def select_safetensors(
    directory: Path, *, layout: str = "auto"
) -> tuple[tuple[Path, ...], dict[str, str] | None]:
    """Select the HF layout, never merge precision variants or alternate exports.

    An index is authoritative. Otherwise model.safetensors is canonical, with a
    single standalone safetensors file accepted for local converted checkpoints.
    Index filenames must be basenames: remote assets cannot escape the snapshot.
    """

    if layout not in {"auto", "canonical", "indexed"}:
        raise AudioWeightsError("weight layout must be auto, canonical or indexed")
    canonical = directory / "model.safetensors"
    if layout == "canonical":
        if not canonical.is_file():
            raise AudioWeightsError("canonical model.safetensors is required")
        return (canonical,), None
    index = directory / "model.safetensors.index.json"
    if layout == "indexed" and not index.is_file():
        raise AudioWeightsError("indexed model.safetensors.index.json is required")
    if index.is_file():
        try:
            value = json.loads(index.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        except (OSError, json.JSONDecodeError) as exc:
            raise AudioWeightsError(f"cannot read checkpoint index: {exc}") from exc
        mapping = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise AudioWeightsError("checkpoint index requires a nonempty weight_map")
        for name, filename in mapping.items():
            if not isinstance(name, str) or not name:
                raise AudioWeightsError("checkpoint tensor names must be nonempty strings")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or "\\" in filename
                or not filename.endswith(".safetensors")
            ):
                raise AudioWeightsError("checkpoint shard names must be safetensors basenames")
        files = tuple(directory / filename for filename in sorted(set(mapping.values())))
        return files, mapping
    if canonical.is_file():
        return (canonical,), None
    files = tuple(sorted(directory.glob("*.safetensors")))
    if len(files) > 1:
        raise AudioWeightsError(
            "ambiguous safetensors files: provide model.safetensors or its index"
        )
    if not files:
        raise AudioWeightsError(
            "safetensors checkpoint is missing; pickle checkpoints are not accepted"
        )
    return files, None


def validate_tensor_index(
    files: tuple[Path, ...],
    mapping: dict[str, str] | None,
    tensor_names: dict[str, tuple[str, ...]],
) -> None:
    """Check names before materializing weights; reject extra/misassigned tensors."""

    seen: set[str] = set()
    for path in files:
        for name in tensor_names[path.name]:
            if name in seen:
                raise AudioWeightsError(f"duplicate checkpoint tensor: {name}")
            seen.add(name)
            if mapping is not None and mapping.get(name) != path.name:
                raise AudioWeightsError(f"tensor is absent from or misassigned by index: {name}")
    if mapping is not None and seen != set(mapping):
        missing = ", ".join(sorted(set(mapping) - seen))
        raise AudioWeightsError(f"checkpoint index tensors are missing: {missing}")


def read_safetensors(directory: Path, *, layout: str = "auto") -> dict[str, Any]:
    """Read only the selected layout, preserving the publisher's tensor dtypes."""
    from safetensors import safe_open

    files, mapping = select_safetensors(directory, layout=layout)
    names = {}
    for path in files:
        with safe_open(path, framework="numpy") as handle:
            names[path.name] = tuple(handle.keys())
    validate_tensor_index(files, mapping, names)
    import mlx.core as mx

    result = {}
    for path in files:
        result.update(mx.load(str(path)))
    return result
