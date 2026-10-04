"""Original TTS provenance is required; public voice files are not consent proof."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def harness(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    runner = runpy.run_path(str(directory / "vibevoice_source_contract.py"))
    monkeypatch.setattr(
        runner["subprocess"],
        "check_output",
        lambda command, **kw: runner["TREE"] if command[-1] == "HEAD^{tree}" else runner["SOURCE"],
    )
    monkeypatch.setattr(runner["subprocess"], "run", lambda *args, **kw: None)
    return runner


def publisher(runner):
    return {
        "sha": runner["SOURCE"],
        "html_url": f"https://github.com/microsoft/VibeVoice/commit/{runner['SOURCE']}",
        "commit": {"tree": {"sha": runner["TREE"]}},
    }


def test_unverified_fork_metadata_is_rejected_before_assets(harness, tmp_path):
    data = publisher(harness)
    data["html_url"] = "https://github.com/unverified/fork/commit/unknown"
    with pytest.raises(ValueError, match="publisher commit"):
        harness["source_contract"](tmp_path, tmp_path, data)


def test_supplemental_tokenizer_revision_must_be_frozen(harness, tmp_path):
    with pytest.raises(ValueError, match="frozen Hub snapshot"):
        harness["source_contract"](tmp_path, tmp_path, publisher(harness))


def test_voice_availability_and_license_never_become_recorded_consent(harness, tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    source, tokenizer = tmp_path / "source", tmp_path / harness["TOKENIZER"]
    source.mkdir()
    tokenizer.mkdir()
    for name in ("README.md", "LICENSE", "pyproject.toml"):
        (source / name).write_text("publisher asset")
    voices = source / "demo/voices"
    voices.mkdir(parents=True)
    (voices / "en-voice.wav").write_bytes(b"available voice fixture")
    special = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|endoftext|>",
    ]
    Tokenizer(WordLevel(dict(zip(special, range(5), strict=True)))).save(
        str(tokenizer / "tokenizer.json")
    )
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}))
    for name in ("vocab.json", "merges.txt", "LICENSE"):
        (tokenizer / name).write_text("tokenizer asset")
    result = harness["source_contract"](source, tokenizer, publisher(harness))
    assert result["status"] == "SOURCE_AND_TOKENIZER_PINNED_NOT_QUALIFIED"
    voice = result["publisher_voice_files"]["demo/voices/en-voice.wav"]
    assert voice["publisher_available"] is True
    assert voice["recorded_consent"] is False
    assert voice["qualification_permission_evidence"].startswith("WIP")
    assert result["qualification"]["production_inference"] == "WIP / not implemented"
