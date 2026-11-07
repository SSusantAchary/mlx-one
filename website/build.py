"""Build and validate the repository-maintained GitHub Pages documentation."""

from __future__ import annotations

import argparse
import shutil
import subprocess

from check import check
from prepare import GENERATED, HERE, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    stage()
    if args.prepare_only:
        return
    binary = shutil.which("zensical")
    if binary is None:
        parser.error("install website/requirements.txt in the active documentation environment")
    subprocess.run([binary, "build", "--clean", "--strict"], cwd=HERE, check=True)
    # Expose provenance as a static artifact without putting it into the search index.
    shutil.copyfile(GENERATED / "source-manifest.json", HERE / "dist/source-manifest.json")
    report = check(HERE / "dist")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
