import json
from types import SimpleNamespace

import pytest

from mlx_one.p0_engine_qualification import digest
from qualification.p01_math import canonical_digest
from qualification.p01_prompts import VERSION, load_prompts, resolve_prompt


def fixture(tmp_path):
    metadata = {"model_id": "fixture", "revision": "a" * 40, "assets": {"weights": "digest"}}
    data = {"schema_version": VERSION, "checkpoint": metadata["model_id"],
        "revision": metadata["revision"], "assets_sha256": canonical_digest(metadata["assets"]),
        "dataset_identity": "local-v1", "provenance": {"origin": "test fixture"},
        "records": [{"id": "one", "text": "actual text", "token_ids": [3, 4],
                     "token_sha256": canonical_digest([3, 4])}]}
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(data))
    return metadata, data, path


@pytest.mark.parametrize("field", ("schema_version", "checkpoint", "revision", "assets_sha256",
                                   "provenance", "dataset_identity"))
def test_incomplete_or_substituted_prompt_identity_rejected(tmp_path, field):
    metadata, data, path = fixture(tmp_path)
    data[field] = "wrong" if field not in {"provenance", "dataset_identity"} else None
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_prompts(path, metadata)


def test_actual_checkpoint_tokens_must_match_frozen_rendered_fixture(tmp_path):
    metadata, _, path = fixture(tmp_path)
    assert resolve_prompt(path, digest(path), metadata, SimpleNamespace(encode=lambda s: [3, 4]),
                          2) == ("actual text", [3, 4])
    with pytest.raises(ValueError, match="tokenization"):
        resolve_prompt(path, digest(path), metadata, SimpleNamespace(encode=lambda s: [3, 5]), 2)
    with pytest.raises(ValueError, match="changed"):
        resolve_prompt(path, "stale", metadata, SimpleNamespace(encode=lambda s: [3, 4]), 2)


def test_duplicate_widths_cannot_silently_choose_different_prompt(tmp_path):
    metadata, data, path = fixture(tmp_path)
    data["records"].append({**data["records"][0], "id": "different"})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        load_prompts(path, metadata)
