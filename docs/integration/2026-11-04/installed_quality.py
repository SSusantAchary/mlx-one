#!/usr/bin/env python3
"""Re-evaluate bounded task quality using only a clean installed native wheel.

Run with -I, fixtures outside the wheel, and a new output path. No data download,
benchmark timing or qualification promotion occurs. Raw evaluation data is local.
"""

import argparse
import importlib.util
import json
import socket
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("fixtures", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
for name in ("torch", "transformers", "onnxruntime", "mlx_lm", "omlx"):
    assert importlib.util.find_spec(name) is None, name


def deny_network(*unused, **kwargs):
    raise RuntimeError("quality replay is offline")


socket.socket.connect = deny_network
import mlx_one  # noqa: E402
from mlx_one import TokenClassificationEngine, ZeroShotImageEngine  # noqa: E402
from mlx_one.engine.modes import resolve_backend, resolved_mode  # noqa: E402
from mlx_one.inference_baseline import TOLERANCES  # noqa: E402
from mlx_one.mlxboost import MLXBoostConfig  # noqa: E402
from mlx_one.server.config import ServerConfig  # noqa: E402

assert "site-packages" in mlx_one.__file__
f = args.fixtures.resolve()
cases = json.loads((f / "runs/m2-20261103/quality-cases.json").read_text())
gold = json.loads((f / "runs/m2-20261103/quality-gold.json").read_text())
reference = json.loads((f / "runs/m2-20261103/quality-hf-tasks.json").read_text())["simple"]
tp = fp = fn = 0
max_error = 0.0
atol, rtol = TOLERANCES["float32"]
with TokenClassificationEngine(f / "checkpoints/bert-ner") as engine:
    for text, truth, ref in zip(cases, gold, reference, strict=True):
        entities = engine.tag(text, aggregation_strategy="simple").predictions[0]["entities"]
        assert len(entities) == len(ref)
        for a, b in zip(entities, ref, strict=True):
            assert {k: v for k, v in a.items() if k != "score"} == {
                k: v for k, v in b.items() if k != "score"
            }
            error = abs(a["score"] - b["score"])
            max_error = max(max_error, error)
            assert error <= atol + rtol * abs(b["score"])
        got = {(e["entity_group"], e["start"], e["end"]) for e in entities}
        expected = {tuple(e) for e in truth}
        tp += len(got & expected)
        fp += len(got - expected)
        fn += len(expected - got)
assert (len(cases), tp, fp, fn) == (100, 153, 156, 45)
ner = dict(
    sentences=len(cases),
    tp=tp,
    fp=fp,
    fn=fn,
    strict_entity_f1=2 * tp / (2 * tp + fp + fn),
    source_decisions_agree=True,
    max_abs_entity_score_error=max_error,
)

matrix = json.loads((f / "runs/m2-20261103/clip/matrix-summary.json").read_text())
clip = {"2": dict(images=0, correct=0), "10": dict(images=0, correct=0)}
with ZeroShotImageEngine(f / "checkpoints/clip-converted") as engine:
    for row in matrix:
        count = row["candidates"]
        if count not in (2, 10):
            continue
        folder = f / "runs/m2-20261103/clip/matrix" / f"{Path(row['image']).stem}-{count}"
        labels = json.loads((folder / "labels.json").read_text())
        result = engine.classify(f / "runs/m2-20261103/clip/real-images" / row["image"], labels)
        top = result.labels[0]["label"]
        assert top == row["hf_top1"]
        clip[str(count)]["images"] += 1
        clip[str(count)]["correct"] += int(top == row["expected_label"])
    try:
        engine.classify(
            f / "runs/m2-20261103/clip/real-images/chelsea.png", [str(i) for i in range(100)]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("100-label profile was admitted")
assert clip == {"2": dict(images=6, correct=6), "10": dict(images=6, correct=5)}

assert resolved_mode(resolve_backend(None, None)) == "reference_dense"
assert not MLXBoostConfig().enabled and not ServerConfig().mlxboost_config.enabled
facts = dict(
    family="qwen2",
    dtype="float16",
    quantization={"bits": 8, "group_size": 64},
    backend="dense",
    slots=1,
    cache_reuse=False,
    mlx_version="0.32.2",
    metal_available=True,
)
assert MLXBoostConfig(fast_decode=True).resolve(**facts)[0].enabled
facts["slots"] = 2
fallback, reason = MLXBoostConfig(fast_decode=True).resolve(**facts)
assert not fallback.enabled and reason
try:
    MLXBoostConfig(fast_decode=True, strict_reference=True).resolve(**facts)
except ValueError:
    pass
else:
    raise AssertionError("strict fallback must raise")
forbidden = sorted(
    n
    for n in sys.modules
    if n.split(".")[0] in {"torch", "transformers", "onnxruntime", "mlx_lm", "omlx"}
)
assert not forbidden
with args.output.open("x") as stream:
    json.dump(
        dict(
            ner=ner,
            clip=clip,
            clip_100_labels_rejected=True,
            mlxboost=dict(
                default_off=True,
                reference_dense_default=True,
                explicit_opt_in=True,
                unsupported_fallback=True,
                strict_raises=True,
            ),
            forbidden_runtime_imports=forbidden,
            wheel_import=True,
        ),
        stream,
        indent=2,
    )
