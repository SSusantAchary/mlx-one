"""Held-out scoring and evidence eligibility without a native backend."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from mlx_one.quality_dataset import (
    DEPTH_BANDS,
    QUALITY_SCHEMA,
    SCORER_VERSION,
    evaluate_text_quality,
    load_quality_manifest,
    paired_logits_metrics,
    quality_coverage,
    retrieval_score,
    summarize_quality_documents,
    validate_text_quality,
)


def manifest(tmp_path):
    data = {"schema_version": QUALITY_SCHEMA, "dataset_id": "fixture-v1", "split": "test",
            "provenance": {"source": "CPU test fixture"}, "license": "CC0",
            "settings": {"scorer": SCORER_VERSION, "context_lengths": [2048],
                         "min_oracle_retrieval_accuracy": .9}}
    for kind, rows in (("documents", [{"id": "a", "text": "Held out text."}]),
                       ("retrieval", [{"id": "a", "context_tokens": 2048,
                           "depth_band": "start", "context": "Fact A", "question": "Fact?",
                           "accepted_answers": ["A"]}])):
        path = tmp_path / f"{kind}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        data[kind] = {"path": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    return path


def test_manifest_requires_offline_integrity_and_held_out_provenance(tmp_path):
    path = manifest(tmp_path)
    loaded = load_quality_manifest(path)
    assert loaded["identity"]["split"] == "test"
    (tmp_path / "documents.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_quality_manifest(path)
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data["split"] = "train"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="held-out"):
        load_quality_manifest(path)


def test_paired_nll_is_shift_invariant_and_token_weighted():
    logits = np.array([[2., 0.], [0., 2.]])
    row = paired_logits_metrics(logits, logits + 1000, (0, 1))
    assert row["oracle_nll_sum"] == pytest.approx(2 * np.log1p(np.exp(-2)))
    assert row["oracle_nll_sum"] == pytest.approx(row["target_nll_sum"])
    other = paired_logits_metrics([[0., 0.]], [[0., 2.]], (0,))
    result = summarize_quality_documents([row, other])
    assert result["scored_tokens"] == 3
    assert result["held_out_nll_mean_delta"] == pytest.approx(
        (other["target_nll_sum"] - other["oracle_nll_sum"]) / 3)
    with pytest.raises(ValueError, match="mapping"):
        paired_logits_metrics(logits, logits, (0,))
    with pytest.raises(ValueError):
        paired_logits_metrics([[np.nan]], [[0.]], (0,))


def test_retrieval_exact_normalization_and_coverage_are_required():
    assert retrieval_score("  ＡＮＳＷＥＲ\n ONE ", ["answer one"])
    assert not retrieval_score("the answer one is somewhere here", ["answer one"])
    documents = {"document_count": 100, "scored_tokens": 100_000}
    rows = [{"depth_band": band, "context_tokens": 2048, "rendered_tokens": 2000}
            for band in DEPTH_BANDS for _ in range(20)]
    assert quality_coverage(documents, {"fixtures": rows}, 2048)["eligible"]
    assert not quality_coverage(documents, {"fixtures": rows[:-1]}, 2048)["eligible"]
    assert not quality_coverage(documents, {"fixtures": rows}, 8192)["eligible"]


def text_fixture():
    data=json.loads((Path(__file__).parent/'alpha_0_1/fixtures/text-quality-v1.json').read_text())
    outputs=[]
    for case in data['cases']:
        expected=case['expected']
        output={'stream_consistent':True,'finish_reason':'stop','tokens':[1]}
        if 'error_type' in expected:
            output.update(error={'type':expected['error_type'],'message':expected['message_contains']},
                nonstream_error={'type':expected['error_type'],'message':expected['message_contains']})
        else:
            output['text']=expected.get('text',json.dumps(expected.get('json')))
        outputs.append(output)
    return data,[deepcopy(outputs) for _ in range(5)]


def test_text_quality_requires_all_frozen_cases_and_five_repetitions():
    data,outputs=text_fixture()
    result=evaluate_text_quality(data,outputs,'a'*64)
    assert result['passed'] and result['coverage_passed'] and result['deterministic_passed']
    assert len(result['checks'])==135
    assert evaluate_text_quality(data,outputs[:1],'a'*64)['coverage_passed'] is None
    shard={**data,'cases':data['cases'][:2]}
    partial=evaluate_text_quality(shard,[row[:2] for row in outputs],'a'*64)
    assert partial['coverage_passed'] is None
    data['text_quality']['required_repetitions']=1
    with pytest.raises(ValueError,match='five-repeat'):
        validate_text_quality(data)


@pytest.mark.parametrize('mutation',['answer','stream','eos','invalid','json_duplicate','json_type','repeat'])
def test_text_quality_retains_failures_and_strict_json(mutation):
    data,outputs=text_fixture()
    if mutation=='answer':
        outputs[0][0]['text']='wrong'
    elif mutation=='stream':
        outputs[0][0]['stream_consistent']=False
    elif mutation=='repeat':
        outputs[0][0]['tokens']=[2]
    else:
        category={'eos':'eos','invalid':'invalid','json_duplicate':'structured','json_type':'structured'}[mutation]
        i=next(i for i,c in enumerate(data['cases']) if c['category']==category)
        if mutation=='eos':
            outputs[0][i]['finish_reason']='length'
        elif mutation=='invalid':
            outputs[0][i]['nonstream_error']['type']='TypeError'
        elif mutation=='json_duplicate':
            outputs[0][i]['text']='{"answer":7,"answer":7}'
        else:
            outputs[0][i]['text']='{"answer":true}'
    assert evaluate_text_quality(data,outputs,'a'*64)['passed'] is False


def test_text_quality_rejects_unknown_rules_and_output_misalignment():
    data,outputs=text_fixture()
    with pytest.raises(ValueError,match='count mismatch'):
        evaluate_text_quality(data,[outputs[0][:-1]],'a'*64)
    data['cases'][0]['expected']={'contains':'Tokyo'}
    with pytest.raises(ValueError,match='scoring rule'):
        validate_text_quality(data)
