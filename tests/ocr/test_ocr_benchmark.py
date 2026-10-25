import copy

import pytest

from benchmarks.ocr.metrics import text_metrics, validate
from benchmarks.ocr.report import render


def record():
    sample = dict(
        latency_seconds=2.0,
        pages=1,
        peak_mlx_memory_gib=1.0,
        peak_process_memory_gib=2.0,
        output_tokens=3,
        transcript="Hello world",
        accuracy=text_metrics("Hello world", "Hello world"),
    )
    return dict(
        schema_version="mlx-one-ocr-benchmark-v1",
        status="MEASURED",
        checkpoint="example/ocr",
        revision="a" * 40,
        precision={"mode": "floating", "dtypes": ["float32"]},
        measurement_type="direct-model",
        fixture_kind="synthetic",
        reference="Hello world",
        task="ocr.recognize",
        workload="OCR-S1",
        hardware=dict(chip="Synthetic test fixture", memory_bytes=32 * 1024**3),
        software=dict(mlx="test"),
        implementation_sha256="b" * 64,
        runner_sha256="c" * 64,
        workload_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
        protocol=dict(batch_size=1, warmups=1, repetitions=5, statistic="median"),
        samples=[copy.deepcopy(sample) for _ in range(5)],
        metrics=dict(latency_seconds=2.0, peak_mlx_memory_gib=1.0, pages_per_second=0.5, cer=0),
    )


def test_accuracy_normalization_and_provenance():
    assert text_metrics("Hello  world", "Hello world")["normalized"]["wer"] == 0
    assert text_metrics("Hello", "hello")["raw"]["cer"] > 0
    value = record()
    validate(value)
    for key in ["revision", "hardware", "implementation_sha256"]:
        broken = copy.deepcopy(value)
        broken.pop(key)
        with pytest.raises(ValueError):
            validate(broken)
    broken = copy.deepcopy(value)
    broken["samples"][0]["transcript"] = "wrong"
    with pytest.raises(ValueError):
        validate(broken)
    broken = copy.deepcopy(value)
    broken["metrics"]["pages_per_second"] = 8
    with pytest.raises(ValueError):
        validate(broken)
    broken = copy.deepcopy(value)
    broken["mock"] = True
    with pytest.raises(ValueError):
        validate(broken)


def test_generator_ignores_mock_and_rejects_duplicate():
    model = dict(
        family="trocr",
        checkpoint="example/ocr",
        revision="a" * 40,
        primary_task="ocr.recognize",
        primary_workload="OCR-S1",
        precision="FP32",
        scope="line",
        name="test",
    )
    value = record()
    assert "Not tested" in render([model], [({**value, "mock": True}, "test.json")])
    assert "**0.500**" in render([model], [(value, "test.json")])
    with pytest.raises(ValueError):
        render([model], [(value, "a.json"), (value, "b.json")])


def test_region_table_and_formula_metric_definitions():
    from benchmarks.ocr.metrics import box_iou, region_metrics, table_metrics

    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    target = [
        {"page": 0, "bbox": [0, 0, 10, 10], "reading_order": 0},
        {"page": 0, "bbox": [20, 20, 30, 30], "reading_order": 1},
    ]
    assert region_metrics(target, target)["f1"] == 1
    reversed_order = [{**target[0], "reading_order": 1}, {**target[1], "reading_order": 0}]
    assert region_metrics(target, reversed_order)["matched_reading_order_pair_accuracy"] == 0
    assert region_metrics(target, [])["recall"] == 0
    table = {"rows": [[{"text": "A", "colspan": 2}]]}
    assert table_metrics(table, table)["cell_span_exact_fraction"] == 1
    assert table_metrics(table, {})["cell_text_exact_fraction"] == 0
