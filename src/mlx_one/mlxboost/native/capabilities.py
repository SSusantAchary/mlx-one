"""Explicit inspection only; importing this module never imports MLX."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import platform
from importlib import metadata, util
from pathlib import Path

from .schemas import MLX_VERSION, NativeKernelUnavailable

ROOT = Path(__file__).parent


def distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "_build").iterdir()):
        if path.is_file() and path.suffix in {".py", ".cpp", ".mm", ".h", ".metal", ".txt"}:
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def mlx_build_sha256():
    spec = util.find_spec("mlx")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(next(iter(spec.submodule_search_locations))) / "lib/libmlx.dylib"
    return file_sha256(path) if path.is_file() else None


def binary_sha256():
    # Bind qualification to all executable assets, not merely the binding .so.
    digest = hashlib.sha256()
    paths = sorted(p for p in ROOT.iterdir() if p.suffix in {".so", ".dylib", ".metallib"})
    if not paths:
        return None
    for path in paths:
        digest.update(path.name.encode() + b"\0" + bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def native_status(*, probe=True):
    """Explicit probe can initialize MLX/Metal; probe=False inspects metadata only.

    Paths in public diagnostics are basenames. None means not probed, not success.
    """
    binary = next(
        (
            ROOT / ("_ext" + suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            if (ROOT / ("_ext" + suffix)).is_file()
        ),
        None,
    )
    status = dict(
        installed=binary is not None,
        loadable=False,
        mlx_version=distribution_version("mlx"),
        metal_available=None,
        reason="extension_unavailable",
        binary_path=binary.name if binary else None,
    )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        status["reason"] = "unsupported_platform"
    elif status["mlx_version"] != MLX_VERSION or distribution_version("mlx-metal") != MLX_VERSION:
        status["reason"] = "mlx_version_mismatch"
    elif binary is not None:
        status["reason"] = "not_probed"
        if probe:
            try:
                module = load_extension()
                status.update(
                    loadable=True,
                    metal_available=True,
                    reason="available_experimental",
                    implementation_sha256=module.implementation_sha256,
                    mlx_build_sha256=module.mlx_build_sha256,
                    binary_sha256=binary_sha256(),
                )
            except NativeKernelUnavailable as exc:
                status["reason"] = exc.reason
    return status


def load_extension():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise NativeKernelUnavailable("unsupported_platform")
    if (
        distribution_version("mlx") != MLX_VERSION
        or distribution_version("mlx-metal") != MLX_VERSION
    ):
        raise NativeKernelUnavailable("mlx_version_mismatch")
    # Inspect build provenance before dlopen, including a same-version MLX rebuild.
    from .registry import read_json

    try:
        build = read_json(ROOT / "_build_info.json")
        if build["mlx_version"] != MLX_VERSION or build["mlx_build_sha256"] != mlx_build_sha256():
            raise NativeKernelUnavailable("mlx_version_mismatch")
        if build["implementation_sha256"] != source_sha256():
            raise NativeKernelUnavailable("stale_binary")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reason = (
            "extension_assets_missing" if any(ROOT.glob("_ext*.so")) else "extension_unavailable"
        )
        raise NativeKernelUnavailable(reason) from exc
    try:
        # MLX registers the array/stream casters in the mlx nanobind domain first.
        mx = importlib.import_module("mlx.core")
        module = importlib.import_module("mlx_one.mlxboost.native._ext")
    except (ImportError, OSError, RuntimeError) as exc:
        raise NativeKernelUnavailable("extension_unavailable") from exc
    if module.mlx_version != MLX_VERSION or module.mlx_build_sha256 != mlx_build_sha256():
        raise NativeKernelUnavailable("mlx_version_mismatch")
    if module.implementation_sha256 != source_sha256():
        raise NativeKernelUnavailable("stale_binary")
    if not mx.metal.is_available():
        raise NativeKernelUnavailable("metal_unavailable")
    if not (ROOT / "mlxboost.metallib").is_file() or not (ROOT / "libmlxboost_ops.dylib").is_file():
        raise NativeKernelUnavailable("extension_assets_missing")
    return module
