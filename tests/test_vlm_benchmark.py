"""Evidence admission and README preservation without publishing synthetic scores."""

import pytest

from benchmarks.vlm.metrics import EVIDENCE, read, validate
from benchmarks.vlm.readme_generator import END, START, records, render, update
from benchmarks.vlm.run import identify
from benchmarks.vlm.workloads import HF_NAMES, workload, workload_metadata


def test_pending_records_have_no_invented_metrics_and_all_targets_are_covered():
    plan, result = records()
    assert len(plan["checkpoints"]) == 5
    legacy = [r for r in result if r["workload"].startswith("VLM-S")]
    assert len(legacy) == 20
    assert len({(r["checkpoint"], r["workload"]) for r in result}) == len(result)
    for record in legacy:
        assert record["status"] == "NOT_RUN"
        assert all(value is None for value in record["metrics"].values())
    assert len(plan["alternatives"]) == 2


def test_readme_generator_is_repeatable_and_preserves_other_content():
    plan, result = records()
    plan = {**plan, "readme_workload": "VLM-S1"}
    text = "Original visual assets\n" + START + "\nstale\n" + END + "\nOriginal text benchmark\n"
    generated = update(text, render(plan, result))
    assert update(generated, render(plan, result)) == generated
    assert generated.startswith("Original visual assets\n")
    assert generated.endswith("\nOriginal text benchmark\n")
    assert generated.count("Pending hardware test") == 5
    with pytest.raises(ValueError, match="marker"):
        update("README with no markers", "replacement")
    with pytest.raises(ValueError, match="marker"):
        update(text + START, "replacement")


@pytest.mark.parametrize(
    "change",
    [
        {"status": "MOCK"},
        {"revision": "unidentified"},
        {"status": "MEASURED", "synthetic": True},
        {"metrics": {"decode_tps": 123.0}},
        {"protocol": "text-128-16"},
    ],
)
def test_unidentified_mock_or_unexecuted_scores_are_rejected(change):
    record = read(EVIDENCE / "results/V01-VLM-S1.json")
    record.update(change)
    with pytest.raises((ValueError, KeyError)):
        validate(record)


def test_benchmark_never_downloads_and_requires_actual_content(tmp_path):
    model = read(EVIDENCE / "benchmark_plan.json")["checkpoints"][0]
    with pytest.raises(ValueError, match="asset mismatch"):
        identify(tmp_path, model)
    for name in ["VLM-S1", "VLM-S2", "VLM-S3", "VLM-S4"]:
        work = workload(name)
        assert len(work["images"]) == (4 if name == "VLM-S3" else 1)
        assert work["source_sizes"][0] == ([1024, 1024] if name == "VLM-S2" else [512, 512])
        assert work["image_sha256"] == workload(name)["image_sha256"]


def test_hf_workloads_remain_distinct_and_reproducible_after_cache_cleanup():
    for name in HF_NAMES:
        metadata = workload_metadata(name)
        assert metadata["image_provenance"]["dataset"] == "huggingface/documentation-images"
        assert len(metadata["image_provenance"]["revision"]) == 40
        assert len(metadata["image_sha256"]) == (4 if name == "VLM-HF-S3" else 1)
        assert metadata["protocol"] == name.lower() + "-v1"
        with pytest.raises(ValueError, match="image-dir"):
            workload(name)


def test_real_measurements_reject_missing_hardware_wrong_images_and_wrong_counts():
    import copy

    _, result = records()
    measured = next((r for r in result if r["status"] == "MEASURED"), None)
    if measured is None:
        pytest.skip("no actual checkpoint measurement has been collected yet")
    for field in ["hardware", "counts", "image_provenance"]:
        record = copy.deepcopy(measured)
        if field == "hardware":
            record["hardware"] = {}
        elif field == "counts":
            record["samples"][0]["output_tokens"] += 1
        else:
            record["image_provenance"]["revision"] = "main"
        with pytest.raises((ValueError, KeyError)):
            validate(record)


def test_image_preparation_verifies_sources_and_padded_derivatives(tmp_path, monkeypatch):
    import hashlib
    import json

    from PIL import Image

    import benchmarks.vlm.prepare_images as preparation

    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "scene.png"
    Image.new("RGB", (4, 2), "red").save(source)
    manifest = {
        "sources": [
            {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ],
        "workloads": {
            "VLM-HF-S1": {
                "images": [{"source": source.name, "file": "prepared.png", "size": [8, 8]}]
            }
        },
    }
    metadata = tmp_path / "manifest.json"
    metadata.write_text(json.dumps(manifest))
    monkeypatch.setattr(preparation, "MANIFEST", metadata)
    output = preparation.prepare(tmp_path / "prepared", source_dir=originals)
    with Image.open(output / "prepared.png") as image:
        assert image.size == (8, 8)
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((4, 4)) == (255, 0, 0)
    manifest["workloads"]["VLM-HF-S1"]["images"][0]["sha256"] = "0" * 64
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prepared image checksum"):
        preparation.prepare(tmp_path / "prepared", source_dir=originals)
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source image checksum"):
        preparation.prepare(tmp_path / "prepared", source_dir=originals)
