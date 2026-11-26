"""Optional, pinned native build; ordinary setuptools builds remain Python-only."""

import os
import platform
import subprocess
from importlib.metadata import version

from setuptools import setup
from setuptools.command.build import build
from setuptools.command.build_py import build_py


class IsolatedBuild(build):
    def initialize_options(self):
        super().initialize_options()
        flavor = "native" if os.environ.get("MLX_ONE_BUILD_NATIVE") == "1" else "standard"
        self.build_base = "build/mlxboost-" + flavor


class PythonBuild(build_py):
    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        # Never reuse binaries left by an earlier editable/native build.
        return [
            p for p in files if not p.endswith((".so", ".dylib", ".metallib", "_build_info.json"))
        ]


kwargs = {"cmdclass": {"build": IsolatedBuild, "build_py": PythonBuild}}
if os.environ.get("MLX_ONE_BUILD_NATIVE") == "1":
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLXBoost native build requires macOS arm64")
    if version("mlx") != "0.32.2" or version("mlx-metal") != "0.32.2":
        raise RuntimeError("Native build requires mlx==mlx-metal==0.32.2 in the build interpreter")
    if subprocess.run(["xcrun", "--find", "metal"], capture_output=True).returncode:
        raise RuntimeError(
            "Metal compiler missing: activate Xcode/Metal toolchain; xcrun --find metal"
        )
    from mlx.extension import CMakeBuild, CMakeExtension

    class NativeBuild(CMakeBuild):
        def finalize_options(self):
            super().finalize_options()
            # MLX 0.32.2 reads a process-wide numeric DEBUG when this is None.
            # Keep unrelated host DEBUG values out of the optional build.
            requested = os.environ.get("MLX_ONE_NATIVE_DEBUG", "0")
            if requested not in {"0", "1"}:
                raise RuntimeError("MLX_ONE_NATIVE_DEBUG must be 0 or 1")
            self.debug = requested == "1"

    kwargs["ext_modules"] = [
        CMakeExtension(
            "mlx_one.mlxboost.native._ext", sourcedir="src/mlx_one/mlxboost/native/_build"
        )
    ]
    kwargs["cmdclass"]["build_ext"] = NativeBuild

setup(**kwargs)
