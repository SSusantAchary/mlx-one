"""Comparator invariants independent of model libraries and GPU availability."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def tooling(monkeypatch, name):
    base = Path(__file__).resolve().parents[2] / "benchmarks/model_parity"
    monkeypatch.syspath_prepend(str(base))
    spec = importlib.util.spec_from_file_location(name, base / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("length", [1, 2, 3, 7, 15, 16, 17, 32, 64, 511, 512, 513, 2047])
def test_chunk_partitions_cover_each_logical_token_once(monkeypatch, length):
    oracle = tooling(monkeypatch, "four_path_oracle")
    parts = oracle.partitions(length)
    assert len(parts) == len(set(parts))
    assert (length,) in parts
    for partition in parts:
        assert min(partition) > 0 and sum(partition) == length
        offset, seen = 0, []
        for width in partition:
            seen.extend(range(offset, offset + width))
            offset += width
        assert seen == list(range(length))


def test_all_six_independent_comparisons_are_required(monkeypatch):
    oracle = tooling(monkeypatch, "four_path_oracle")
    assert {frozenset(pair) for pair in oracle.PAIRS} == {
        frozenset(pair) for pair in ("AB", "AC", "AD", "BC", "BD", "CD")
    }


def test_recurrent_reference_cache_does_not_invent_absolute_offset(monkeypatch):
    from types import SimpleNamespace

    oracle = tooling(monkeypatch, "four_path_oracle")
    assert oracle.cache_offsets([SimpleNamespace(offset=17), SimpleNamespace(cache=[])]) == [
        17,
        None,
    ]


@pytest.mark.parametrize(
    "family,name,axis",
    [
        ("qwen3", "model.layers.0.self_attn.q_norm#1", 1),
        ("lfm2", "model.layers.0.self_attn.k_layernorm#1", 1),
        ("qwen3_5", "language_model.layers.3.self_attn.q_norm#1", 1),
        ("qwen3_5", "language_model.layers.0.linear_attn.norm#1", 1),
        ("gemma3_text", "model.layers.0.self_attn.q_norm#1", 2),
        ("phi3", "model.layers.0.self_attn.rope#1", 2),
        ("llama", "model.layers.0.self_attn.rope#1", 2),
        ("qwen2", "model.layers.0.self_attn.rope#1", 2),
    ],
)
def test_trace_slices_tokens_instead_of_heads(monkeypatch, family, name, axis):
    oracle = tooling(monkeypatch, "four_path_oracle")
    values = np.arange(1 * 3 * 5 * 7).reshape(1, 3, 5, 7)
    actual = oracle.last_token_event(name, values, family)
    np.testing.assert_array_equal(actual, np.take(values, [-1], axis=axis))


@pytest.mark.parametrize("offset,retained", [(510, 510), (511, 511), (512, 511), (513, 511)])
def test_sliding_reference_mask_tracks_returned_kv_width(monkeypatch, offset, retained):
    from types import SimpleNamespace

    control = tooling(monkeypatch, "window_boundary_control")
    assert (
        control.reference_retained_before_single_query(SimpleNamespace(max_size=512), offset)
        == retained
    )
    assert control.reference_retained_before_single_query(SimpleNamespace(), offset) == offset


def test_frozen_gate_rejects_matching_argmax_with_other_logit_failures(monkeypatch):
    oracle = tooling(monkeypatch, "four_path_oracle")
    reference = np.array([[[10.0, 0.0, 1.0]]], dtype=np.float32)
    candidate = np.array([[[10.0, 0.2, 1.0]]], dtype=np.float32)
    assert reference.argmax() == candidate.argmax()
    assert oracle.compare(reference, candidate, "bfloat16")["status"] == "FAIL"


@pytest.mark.parametrize(
    "kind,text,finish,expected",
    [
        ("json", '{"name":', "length", "TRUNCATED"),
        ("json", '```json\n{"name":"Lin","count":2}\n```', "stop", "FAIL_JSON"),
        ("json", '{"name":"Lin","count":2}', "length", "PASS"),
        ("arithmetic", "17 + 25 =", "length", "TRUNCATED"),
        ("arithmetic", "43", "stop", "FAIL_ARITHMETIC"),
        ("arithmetic", "17 + 25 = 42", "stop", "FAIL_INSTRUCTION"),
        ("arithmetic", "42", "stop", "PASS"),
    ],
)
def test_task_truncation_does_not_change_expected_answers(
    monkeypatch, kind, text, finish, expected
):
    quality = tooling(monkeypatch, "quality")
    case = {"id": kind, "kind": "json" if kind == "json" else "exact", "expected": "42"}
    assert quality.classify(case, text, finish) == expected


def test_task_budget_revision_preserves_prompts_and_answers():
    base = Path(__file__).resolve().parents[2] / "scripts/qualification"
    old = json.loads((base / "ten_model_quality_v1.json").read_text())
    new = json.loads((base / "ten_model_quality_v2.json").read_text())
    assert [{k: v for k, v in case.items() if k != "max_tokens"} for case in new["cases"]] == (
        old["cases"]
    )
    budgets = {case["id"]: case["max_tokens"] for case in new["cases"]}
    assert budgets["json"] == 128 and budgets["arithmetic"] == 256


def test_retrieval_format_failure_is_distinct_from_wrong_value(monkeypatch):
    quality = tooling(monkeypatch, "quality")
    case = {"id": "bounded-retrieval", "kind": "exact", "expected": "orchid-17"}
    assert quality.classify(case, 'The secret value is "orchid-17".', "stop") == "FAIL_INSTRUCTION"
    assert quality.classify(case, "background", "stop") == "FAIL_RETRIEVAL"
    assert quality.classify(case, "orchid-17", "stop") == "PASS"
