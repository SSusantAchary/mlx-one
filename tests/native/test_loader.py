"""ABI/source guard must run before importing the binary or initializing Metal."""

import json

import pytest

from mlx_one.mlxboost.native import capabilities as cap
from mlx_one.mlxboost.native.schemas import NativeKernelUnavailable


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"mlx_version": "0.33.0"}, "mlx_version_mismatch"),
        ({"mlx_build_sha256": "wrong"}, "mlx_version_mismatch"),
        ({"implementation_sha256": "wrong"}, "stale_binary"),
    ],
)
def test_mismatched_build_does_not_dlopen_or_initialize_metal(
    tmp_path, monkeypatch, mutation, reason
):
    build = dict(mlx_version="0.32.2", mlx_build_sha256="mlx", implementation_sha256="source")
    build.update(mutation)
    (tmp_path / "_build_info.json").write_text(json.dumps(build))
    monkeypatch.setattr(cap, "ROOT", tmp_path)
    monkeypatch.setattr(cap.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cap.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cap, "distribution_version", lambda _: "0.32.2")
    monkeypatch.setattr(cap, "mlx_build_sha256", lambda: "mlx")
    monkeypatch.setattr(cap, "source_sha256", lambda: "source")

    def forbidden(*args):
        pytest.fail("binary or Metal imported before compatibility guard")

    monkeypatch.setattr(cap.importlib, "import_module", forbidden)
    with pytest.raises(NativeKernelUnavailable, match=reason):
        cap.load_extension()


def test_installed_mlx_version_rejected_before_any_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "ROOT", tmp_path)
    monkeypatch.setattr(cap.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cap.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cap, "distribution_version", lambda _: "0.33.0")
    with pytest.raises(NativeKernelUnavailable, match="mlx_version_mismatch"):
        cap.load_extension()
