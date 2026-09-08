"""Privacy-safe software provenance for reproducible lifecycle runs."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from mlx_one.schemas import SoftwareProvenance


def collect_software_provenance() -> SoftwareProvenance:
    """Collect public runtime versions without usernames or private paths."""
    packages = {}
    for name in ("mlx", "mlx-lm", "mlx-one"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return SoftwareProvenance(
        python_version=platform.python_version(),
        packages=packages,
        os_version=platform.platform(terse=True),
        metadata={"implementation": sys.implementation.name},
    )
