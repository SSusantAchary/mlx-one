import ast
from importlib import metadata
from pathlib import Path

from mlx_one.core.compatibility import (
    MLXCompatibilityStatus,
    get_mlx_compatibility,
    load_mlx_compatibility_manifest,
)


def test_packaged_manifest_has_truthful_five_release_window() -> None:
    manifest = load_mlx_compatibility_manifest()

    assert manifest["latest_upstream"] == "0.32.2"
    assert manifest["latest_verified"] is None
    assert tuple(manifest["mlx_versions"]) == (
        "0.32.2",
        "0.32.1",
        "0.32.0",
        "0.31.2",
        "0.31.1",
    )
    assert {item["status"] for item in manifest["mlx_versions"].values()} == {
        "supported"
    }


def test_offline_compatibility_resolution() -> None:
    assert get_mlx_compatibility(runtime_version="0.32.1").state is (
        MLXCompatibilityStatus.SUPPORTED
    )
    assert get_mlx_compatibility(runtime_version="0.33.0").state is (
        MLXCompatibilityStatus.UNVERIFIED
    )
    assert get_mlx_compatibility(runtime_version="0.33.0rc1").state is (
        MLXCompatibilityStatus.UNVERIFIED
    )
    assert get_mlx_compatibility(runtime_version="0.31.3").state is (
        MLXCompatibilityStatus.UNVERIFIED
    )
    assert get_mlx_compatibility(runtime_version="0.30.0").state is (
        MLXCompatibilityStatus.LEGACY
    )


def test_missing_or_unimportable_mlx_is_incompatible(monkeypatch) -> None:
    from mlx_one.core import compatibility

    def missing(_name: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(compatibility.metadata, "version", missing)
    assert get_mlx_compatibility().state is MLXCompatibilityStatus.INCOMPATIBLE

    monkeypatch.setattr(compatibility.metadata, "version", lambda _name: "0.32.2")
    monkeypatch.setattr(compatibility, "_mlx_importable", lambda: False)
    result = get_mlx_compatibility()
    assert result.state is MLXCompatibilityStatus.INCOMPATIBLE
    assert "cannot be imported" in result.reason


def test_runtime_modules_do_not_import_ecosystem_backends() -> None:
    root = Path(__file__).parents[1] / "src" / "mlx_one"
    forbidden = {"mlx_lm", "mlx_vlm", "mlx_audio", "mlx_embeddings"}
    found = []
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {item.name.split(".", 1)[0] for item in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".", 1)[0]}
            else:
                continue
            if names & forbidden:
                found.append(str(source.relative_to(root)))
    assert found == []
