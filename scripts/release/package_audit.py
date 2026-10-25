"""Inspect exact distribution payloads for source identity, notices and private assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath


def sha(value):
    return hashlib.sha256(value).hexdigest()


def audit(root, dist):
    records = []
    expected = {
        str(p.relative_to(root / "src")): sha(p.read_bytes())
        for p in (root / "src/mlx_one").rglob("*.py")
    }
    expected_upstream_licenses = {
        str(p.relative_to(root / "src")): sha(p.read_bytes())
        for p in (root / "src/mlx_one").rglob("UPSTREAM_LICENSE")
    }
    for path in sorted(dist.iterdir()):
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                payload = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
        elif path.name.endswith(".tar.gz"):
            with tarfile.open(path) as archive:
                payload = {
                    m.name: archive.extractfile(m).read()
                    for m in archive.getmembers()
                    if m.isfile()
                }
        else:
            continue
        bad_paths, sensitive, forbidden = [], [], []
        for name, data in payload.items():
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                bad_paths.append(name)
            if any(
                part in {"references_libs", "qualification", ".git", ".venv", "__pycache__"}
                for part in pure.parts
            ) or name.endswith((".safetensors", ".pyc", ".DS_Store")):
                forbidden.append(name)
            if re.search(
                rb"/Users/[^/\s]+/|/home/[^/\s]+/|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}", data
            ):
                sensitive.append(name)
        actual = {}
        for name, data in payload.items():
            position = name.find("mlx_one/")
            if position >= 0 and name.endswith(".py"):
                actual[name[position:]] = sha(data)
        notices = [n for n in payload if n.endswith("UPSTREAM_LICENSE")]
        actual_upstream_licenses = {
            name[name.index("mlx_one/"):]: sha(payload[name])
            for name in notices if "mlx_one/" in name
        }
        tq3_notices = [n for n in payload if n.endswith("quantization/turboquant/NOTICE.txt")]
        ui_notices = [n for n in payload if n.endswith("ui/dist/THIRD_PARTY_NOTICES.txt")]
        derived_notices = [n for n in payload if PurePosixPath(n).name in {"NOTICE", "NOTICE.md"}]
        ui_prefix = "src/mlx_one/ui/dist/"
        ui_sources = {
            str(p.relative_to(root / "src")): sha(p.read_bytes())
            for p in (root / ui_prefix).rglob("*") if p.is_file()
        }
        actual_ui = {
            n[n.index("mlx_one/ui/dist/"):]: sha(data)
            for n, data in payload.items() if "mlx_one/ui/dist/" in n
        }
        apache = [
            n
            for n, data in payload.items()
            if n.endswith(("LICENSE", "LICENSE.txt")) and b"Apache License" in data
        ]
        record = {
            "filename": path.name,
            "sha256": sha(path.read_bytes()),
            "bytes": path.stat().st_size,
            "files": {n: sha(data) for n, data in payload.items()},
            "production_python_files": len(actual),
            "exact_source_match": actual == expected,
            "unsafe_paths": bad_paths,
            "private_paths_or_token_patterns": sensitive,
            "forbidden_payloads": forbidden,
            "mit_notices": notices,
            "exact_upstream_license_match": actual_upstream_licenses == expected_upstream_licenses,
            "ui_notices": ui_notices,
            "tq3_notices": tq3_notices,
            "derived_source_notices": derived_notices,
            "exact_ui_match": actual_ui == ui_sources,
            "apache_notices": apache,
        }
        record["status"] = (
            "PASS"
            if actual == expected
            and notices
            and actual_upstream_licenses == expected_upstream_licenses
            and any(n.endswith("models/vision_language/UPSTREAM_LICENSE") for n in notices)
            and len(ui_notices) == 1
            and len(tq3_notices) == 1
            and len(derived_notices) == 4
            and actual_ui == ui_sources
            and apache
            and not (bad_paths or sensitive or forbidden)
            else "FAIL"
        )
        records.append(record)
    return {
        "timestamp": time.time(),
        "artifacts": records,
        "status": "PASS"
        if len(records) == 2 and all(r["status"] == "PASS" for r in records)
        else "FAIL",
        "scope": "Payload and license notice inspection; not a vulnerability database audit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = audit(Path(__file__).resolve().parents[2], args.dist)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
