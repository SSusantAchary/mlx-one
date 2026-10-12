"""Network-free acceptance-gate and pinned streaming-wrapper behavior checks."""

from __future__ import annotations

import json
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    return runpy.run_path(str(directory / "qwen_audio_qualification.py"))


@pytest.mark.parametrize(
    "field", ("model", "revision", "reference_source_commit", "fixture_sha256")
)
def test_layer_probe_rejects_changed_reference_contract_before_backend(
    monkeypatch, tmp_path, field
):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    probe = runpy.run_path(str(directory / "qwen_audio_layer_probe.py"))
    fixture = tmp_path / "fixture.npz"
    fixture.write_bytes(b"network-free checksum fixture")
    metadata = {
        "model": probe["MODEL"],
        "revision": probe["PINS"][probe["MODEL"]],
        "reference_source_commit": probe["SOURCE"],
        "fixture_sha256": probe["digest"](fixture),
    }
    probe["validate_reference"](metadata, fixture)
    metadata[field] = "changed"
    with pytest.raises(ValueError, match="frozen manifest"):
        probe["validate_reference"](metadata, fixture)


def artifacts(tmp_path, *, count=100, task="transcribe", mode="offline"):
    common = {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "revision": "pin",
        "task": task,
        "mode": mode,
        "dtype": "bfloat16",
        "reference_source_commit": "pin",
        "attention": "sdpa-global",
        "manifest_sha256": "hash",
        "language_subset": ["English"],
        "peak_process_bytes": 1024,
        "cleanup_growth_bytes": 0,
        "predictions": [
            {
                "id": str(i),
                "anchor": i < 16,
                "reference": "one two",
                "prediction": "One two.",
                "units": [{"text": "one", "start": 0.0, "end": 0.08}],
                "events": [
                    {
                        "kind": "final",
                        "text": "One two.",
                        "language": "English",
                        "audio_seconds": 2.0,
                        "sequence": 0,
                    }
                ],
            }
            for i in range(count)
        ],
    }
    ref, native = tmp_path / "reference.json", tmp_path / "native.json"
    ref.write_text(json.dumps({"backend": "reference", **common}))
    native.write_text(json.dumps({"backend": "native", **common}))
    return SimpleNamespace(reference=ref, native=native)


def change_native(args, mutate):
    payload = json.loads(args.native.read_text())
    mutate(payload)
    args.native.write_text(json.dumps(payload))


def test_full_scoped_gate_and_insufficient_anchor_run(runner, tmp_path):
    args = artifacts(tmp_path)
    assert runner["compare"](args)["status"] == "PASS"
    change_native(args, lambda p: p.update(cleanup_growth_bytes=65 * 1024**2))
    assert not runner["compare"](args)["gates"]["cleanup"]
    assert runner["compare"](artifacts(tmp_path, count=16))["status"] == "FAIL"


def test_exact_anchor_gate_preserves_punctuation(runner, tmp_path):
    args = artifacts(tmp_path)
    change_native(args, lambda p: p["predictions"][0].update(prediction="One two;"))
    result = runner["compare"](args)
    assert result["status"] == "FAIL" and result["mismatches"] == ["0"]
    assert result["gates"]["wer_degradation"]


def test_comparison_identifies_validation_only_candidate(runner, tmp_path):
    args = artifacts(tmp_path)
    change_native(args, lambda p: p.update(numerical_candidate="cpu-bf16-rounding"))
    assert runner["compare"](args)["numerical_candidate"] == "cpu-bf16-rounding"


@pytest.mark.parametrize(
    "flag", ("--numerical-candidate", "--text-math-candidate", "--layernorm-candidate")
)
def test_reference_run_cannot_enable_native_candidate(runner, monkeypatch, tmp_path, capsys, flag):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen_audio_qualification.py",
            "run",
            "--backend",
            "reference",
            "--model",
            "Qwen/Qwen3-ASR-0.6B",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "reference.json"),
            flag,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runner["main"]()
    assert exc.value.code == 2
    assert "never the reference" in capsys.readouterr().err
    assert not (tmp_path / "reference.json").exists()


@pytest.mark.parametrize(
    "field", ("checkpoint_model", "checkpoint_revision", "reference_source_commit")
)
def test_token_probe_requires_frozen_contract_before_native_import(monkeypatch, field):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    probe = runpy.run_path(str(directory / "qwen_audio_probe.py"))
    model = "Qwen/Qwen3-ASR-1.7B"
    fixture = {
        "checkpoint_model": model,
        "checkpoint_revision": probe["PINS"][model],
        "reference_source_commit": probe["SOURCE"],
    }
    assert probe["validate_token_fixture"](fixture) == model
    fixture[field] = "changed"
    with pytest.raises(ValueError, match="frozen checkpoint/source"):
        probe["validate_token_fixture"](fixture)
    fixture.pop(field)
    with pytest.raises(ValueError, match="lacks.*contract"):
        probe["validate_token_fixture"](fixture)


def test_qualification_rejects_combined_experiments_before_backend(
    runner, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen_audio_qualification.py",
            "run",
            "--backend",
            "native",
            "--model",
            "Qwen/Qwen3-ASR-1.7B",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "out.json"),
            "--numerical-candidate",
            "--text-math-candidate",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runner["main"]()
    assert exc.value.code == 2
    assert "one isolated numerical candidate" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_token_diagnostic_records_ties_without_reclassifying_a_failure(monkeypatch):
    import numpy as np

    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    probe = runpy.run_path(str(directory / "qwen_audio_probe.py"))
    scores = np.full(151936, -10.0, dtype=np.float32)
    scores[[13, 1335]] = 40.0
    reference = probe["score_decision"](scores)
    assert reference["argmax"] == 13 and reference["margin"] == 0.0
    assert reference["tied_maximum_ids"] == [13, 1335]
    scores[1335] = 40.25
    native = probe["score_decision"](scores)
    assert native["argmax"] == 1335 and native["margin"] == 0.25
    assert native["argmax"] != reference["argmax"]
    assert "status" not in native  # Numerical diagnostics cannot qualify a model.
    scores[0] = float("nan")
    with pytest.raises(ValueError, match="finite vocabulary"):
        probe["score_decision"](scores)


@pytest.mark.parametrize(
    "field", ("model", "revision", "reference_source_commit", "fixture_sha256")
)
def test_classifier_probe_rejects_changed_contract_before_native_import(
    monkeypatch, tmp_path, field
):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    probe = runpy.run_path(str(directory / "qwen_alignment_classifier_probe.py"))
    fixture = tmp_path / "fixture.npz"
    fixture.write_bytes(b"classifier checksum test")
    metadata = {
        "model": probe["MODEL"],
        "revision": probe["PINS"][probe["MODEL"]],
        "reference_source_commit": probe["SOURCE"],
        "fixture_sha256": probe["digest"](fixture),
    }
    probe["validate_operator_metadata"](metadata, fixture)
    metadata[field] = "changed"
    with pytest.raises(ValueError, match="frozen contract"):
        probe["validate_operator_metadata"](metadata, fixture)


@pytest.mark.parametrize("boundary", [0.24, float("nan"), float("inf"), -0.08])
def test_bad_alignment_boundaries_never_qualify(runner, tmp_path, boundary):
    args = artifacts(tmp_path, task="align")
    change_native(args, lambda p: p["predictions"][0]["units"][0].update(end=boundary))
    assert runner["compare"](args)["status"] == "FAIL"


def test_streaming_compares_complete_hypotheses_not_only_final_text(runner, tmp_path):
    args = artifacts(tmp_path, mode="stream")
    change_native(args, lambda p: p["predictions"][0]["events"][0].update(text="two."))
    assert not runner["compare"](args)["gates"]["complete_hypothesis_events"]
    change_native(args, lambda p: p.update(mode="offline"))
    with pytest.raises(ValueError, match="mode"):
        runner["compare"](args)


def test_reference_wrapper_chunking_and_distinct_finish_rollback(monkeypatch):
    path = Path(__file__).parents[1] / "benchmarks/qwen_audio_stream_reference.py"
    runner = runpy.run_path(str(path))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16", inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.inference.utils",
        SimpleNamespace(parse_asr_output=lambda raw, user_language: (user_language, raw)),
    )
    calls = []

    class Processor:
        tokenizer = SimpleNamespace(encode=list, decode=lambda ids, **kw: "".join(ids))

        def __call__(self, *, text, audio, **kw):
            calls.append((text, len(audio)))
            batch = SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 1)))
            return SimpleNamespace(to=lambda **kw: {"input_ids": batch.input_ids})

    # Match the processor's mapping-and-attribute interface without tensor imports.
    class Batch(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, **kw):
            return self

    class MappingProcessor(Processor):
        def __call__(self, *, text, audio, **kw):
            calls.append((text, len(audio)))
            return Batch(input_ids=SimpleNamespace(shape=(1, 1)))

    class Sequences:
        def __getitem__(self, key):
            return "abcdefghij"

    model = SimpleNamespace(
        processor=MappingProcessor(),
        _build_text_prompt=lambda context, language: "prompt:",
        model=SimpleNamespace(generate=lambda **kw: SimpleNamespace(sequences=Sequences())),
    )
    events = runner["reference_stream"](model, [0.0] * 96017, "English", 17)
    assert calls == [
        ("prompt:", 32000),
        ("prompt:", 64000),
        ("prompt:abcde", 96000),
        ("prompt:abcdeabcde", 96017),
    ]
    assert [event["kind"] for event in events] == ["partial"] * 3 + ["final"]
    assert events[-1]["text"] == "abcdeabcdeabcdefghij"
    assert events[-1]["audio_seconds"] == 96017 / 16000
    with pytest.raises(ValueError, match="30 seconds"):
        runner["reference_stream"](model, [0.0] * 480001, "English", 32000)
