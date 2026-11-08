"""The release audit checks notice and metadata contents rather than a fixed count."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/release/package_audit.py"
SPEC = importlib.util.spec_from_file_location("package_payload_audit", PATH)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def artifacts(root, *, changed=None, missing=None):
    package = root / "src/mlx_one"
    files = {
        "__init__.py": b'__version__ = "0.1.1"\n',
        "data/scope.json": b'{"release_approval":"PENDING"}',
        "models/UPSTREAM_LICENSE": b"MIT License",
        "models/vision_language/UPSTREAM_LICENSE": b"MIT License",
        "quantization/turboquant/NOTICE.txt": b"notice",
        "quantization/turboquant/LICENSE.txt": b"Apache License",
        "ui/dist/index.html": b"candidate UI",
        "ui/dist/THIRD_PARTY_NOTICES.txt": b"UI attribution",
        **{f"models/family{i}/NOTICE.md": f"attribution {i}".encode() for i in range(7)},
    }
    for name, data in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "pyproject.toml").write_text(
        '[tool.setuptools.package-data]\nmlx_one = ["data/*.json", "models/**/UPSTREAM_LICENSE", '
        '"models/**/NOTICE.md", "quantization/turboquant/*.txt", "ui/dist/**/*"]\n'
    )
    payload = dict(files)
    if changed:
        payload[changed] = b"changed package data"
    if missing:
        del payload[missing]
    dist = root / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "candidate.whl", "w") as archive:
        for name, data in payload.items():
            archive.writestr("mlx_one/" + name, data)
    with tarfile.open(dist / "candidate.tar.gz", "w:gz") as archive:
        for name, data in payload.items():
            member = tarfile.TarInfo("candidate/src/mlx_one/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return dist


def test_seven_exact_notices_pass_without_a_hard_coded_notice_count(tmp_path):
    result = audit_module.audit(tmp_path, artifacts(tmp_path))
    assert result["status"] == "PASS"
    assert all(a["exact_derived_notice_match"] for a in result["artifacts"])
    assert all(a["exact_package_data_match"] for a in result["artifacts"])


@pytest.mark.parametrize("changed", ["models/family0/NOTICE.md", "data/scope.json"])
def test_changed_notice_or_support_metadata_fails_even_with_same_file_count(tmp_path, changed):
    result = audit_module.audit(tmp_path, artifacts(tmp_path, changed=changed))
    assert result["status"] == "FAIL"
    assert all("mlx_one/" + changed in a["changed_package_data"] for a in result["artifacts"])


def test_missing_notice_fails(tmp_path):
    result = audit_module.audit(tmp_path, artifacts(tmp_path, missing="models/family0/NOTICE.md"))
    assert result["status"] == "FAIL"
    assert all(not a["exact_derived_notice_match"] for a in result["artifacts"])
