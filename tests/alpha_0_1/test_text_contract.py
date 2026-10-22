"""Early loader rejection, including genuine CPU non-finite payload validation."""

import json
import struct
import sys

import pytest

from mlx_one.text.loading import (
    TextModelLoadError,
    _checkpoint_files,
    _load_safetensor_file,
    _quantization_config,
)


@pytest.mark.parametrize("metadata", [False, 0, [], "q4", {"bits": 4.0}, {"bits": True}])
def test_invalid_quantization_cannot_become_float_or_affine(metadata):
    with pytest.raises(ValueError, match="object|unsupported.*bits"):
        _quantization_config({"quantization": metadata})


@pytest.mark.parametrize("indexed", [True, False])
def test_local_shard_symlink_cannot_escape_checkpoint(tmp_path, indexed):
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"not model weights")
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "model.safetensors").symlink_to(outside)
    if indexed:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    with pytest.raises(TextModelLoadError, match="unsafe shard"):
        _checkpoint_files(root)


def test_pinned_hub_blob_links_remain_compatible(tmp_path):
    repo = tmp_path / "models--org--model"
    root = repo / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    (repo / "blobs").mkdir()
    blob = repo / "blobs" / ("b" * 64)
    blob.write_bytes(b"fixture")
    shard = root / "model.safetensors"
    shard.symlink_to(blob)
    assert _checkpoint_files(root) == ((shard,), None)


def test_nonfinite_payload_fails_before_mlx_import(tmp_path, monkeypatch):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1],
                                  "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", float("nan")))
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(ValueError, match="non-finite checkpoint"):
        _load_safetensor_file(path)
