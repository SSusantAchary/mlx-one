#!/usr/bin/env python3
"""Replay existing source parity, bounded quality and wheel interface checks.

Requires a clean checkout, installed native-only wheel and pinned local fixtures.
Reference NPZ/JSON comes from sealed M1-M3 runs; no reference runtime executes.
All outputs go to a new directory. Existing runs are never overwritten.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkout", required=True, type=Path)
p.add_argument("--fixtures", required=True, type=Path)
p.add_argument("--python", required=True, type=Path, help="native-only installed wheel interpreter")
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
root, fixtures, output = (x.resolve() for x in (a.checkout, a.fixtures, a.output))
# Preserve the venv executable symlink; resolving it selects the system interpreter.
python = a.python.absolute()
output.mkdir(parents=True, exist_ok=False)
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
env = {**os.environ, "MLX_ENABLE_TF32": "0", "HF_HUB_OFFLINE": "1"}
env.pop("PYTHONPATH", None)
commands = []
bench = root / "benchmarks/native_tasks"


def run(name, script, *args):
    command = [str(python), "-I", str(script), *map(str, args)]
    commands.append(dict(name=name, argv=command))
    with (
        (output / (name + ".stdout")).open("x") as stdout,
        (output / (name + ".stderr")).open("x") as stderr,
    ):
        subprocess.run(
            command, cwd="/private/tmp", env=env, stdout=stdout, stderr=stderr, check=True
        )
    print("PASS", name, flush=True)


def compare(name, reference, actual):
    report = output / (name + ".json")
    run(
        name,
        bench / "compare_tensors.py",
        reference,
        actual,
        "--dtype",
        "float32",
        "--output",
        report,
    )
    d = json.loads(report.read_text())
    assert d["passed"]
    return dict(case=name, tensors=len(d["tensors"]), passed=True, reference_sha256=sha(reference))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


parity = []
m1, m2, m3 = [fixtures / "runs" / f"m{n}-20261103" for n in (1, 2, 3)]
for maximum, tag in [(32, "max32"), (512, "b2")]:
    actual = output / ("distilbert-" + tag + ".npz")
    run(
        "distilbert-" + tag,
        bench / "export_distilbert_mlx.py",
        fixtures / "checkpoints/distilbert-sst2",
        m1 / "expanded-cases.json",
        actual,
        "--batch-size",
        2,
        "--max-length",
        maximum,
    )
    parity.append(compare("distilbert-" + tag + "-parity", m1 / f"expanded-{tag}-hf.npz", actual))
for batch, ref in [(1, "ner-b1-hf.npz"), (2, "ner-b2-hf-final.npz")]:
    actual, tasks = output / f"ner-b{batch}.npz", output / f"ner-b{batch}-tasks.json"
    run(
        f"ner-b{batch}",
        bench / "export_ner.py",
        "native",
        fixtures / "checkpoints/bert-ner",
        bench / "ner_cases.json",
        actual,
        tasks,
        "--batch-size",
        batch,
    )
    parity.append(compare(f"ner-b{batch}-parity", m2 / ref, actual))
for row in json.loads((m2 / "clip/matrix-summary.json").read_text()):
    if row["candidates"] not in (2, 10):
        continue
    name = f"clip-{Path(row['image']).stem}-{row['candidates']}"
    folder = m2 / "clip/matrix" / f"{Path(row['image']).stem}-{row['candidates']}"
    actual = output / (name + ".npz")
    run(
        name,
        bench / "export_clip.py",
        "native",
        fixtures / "checkpoints/clip-converted",
        m2 / "clip/real-images" / row["image"],
        folder / "labels.json",
        actual,
    )
    parity.append(compare(name + "-parity", folder / "reference.npz", actual))
actual = output / "silero-full.npz"
run(
    "silero-full",
    bench / "export_silero.py",
    "native",
    fixtures / "checkpoints/silero-vad",
    fixtures / "checkpoints/silero-vad/test.wav",
    actual,
)
parity.append(compare("silero-full-parity", m3 / "silero-full-reference.npz", actual))
run(
    "sentiment-quality",
    bench / "evaluate_sentiment.py",
    "native",
    fixtures / "checkpoints/distilbert-sst2",
    m1 / "sst2-cases.json",
    output / "sentiment-quality.json",
)
sentiment = json.loads((output / "sentiment-quality.json").read_text())
reference = json.loads((m1 / "quality-hf.json").read_text())
assert sentiment["count"] == 872 and sentiment["correct"] == 794
assert all(
    x["prediction"] == y["prediction"]
    for x, y in zip(sentiment["rows"], reference["rows"], strict=True)
)
run(
    "ner-clip-quality",
    Path(__file__).with_name("installed_quality.py"),
    fixtures,
    output / "ner-clip-quality.json",
)
# Make a new portable path view of the frozen VAD fixture list; preserve its bytes.
vad_fixtures = json.loads((m3 / "quality/fixtures.json").read_text())
for item in vad_fixtures["fixtures"]:
    item["path"] = str(m3 / "quality" / Path(item["path"]).name)
view = output / "vad-fixtures.json"
view.write_text(json.dumps(vad_fixtures, indent=2) + "\n")
run(
    "vad-quality",
    bench / "evaluate_vad.py",
    "native",
    fixtures / "checkpoints/silero-vad",
    view,
    output / "vad-quality.json",
)
vad = json.loads((output / "vad-quality.json").read_text())
vad_ref = json.loads((m3 / "quality/reference.json").read_text())
assert [vad[k] for k in ["tp", "fp", "fn", "tn"]] == [10, 0, 0, 3]
assert all(
    x["segments"] == y["segments"] for x, y in zip(vad["rows"], vad_ref["rows"], strict=True)
)
run(
    "sentiment-wheel",
    bench / "clean_classification_smoke.py",
    fixtures / "checkpoints/distilbert-sst2",
)
run(
    "native-wheel",
    bench / "clean_m2_smoke.py",
    fixtures / "checkpoints/bert-ner",
    fixtures / "checkpoints/clip-converted",
    m2 / "clip/real-images/chelsea.png",
    fixtures / "checkpoints/silero-vad",
)
quality = json.loads((output / "ner-clip-quality.json").read_text())
summary = dict(
    parity=parity,
    parity_cases=len(parity),
    tensors=sum(x["tensors"] for x in parity),
    sentiment=dict(count=872, correct=794, accuracy=794 / 872, source_top1_agreement=872),
    **quality,
    vad=dict(tp=10, fp=0, fn=0, tn=3, source_segments_agree=True),
    interfaces=dict(
        sentiment="Python/CLI/HTTP PASS",
        ner="Python/CLI/HTTP PASS",
        clip="Python/CLI/HTTP PASS",
        silero="Python/CLI/HTTP PASS",
    ),
    qualification_promotions=0,
    reference_execution="none; sealed source oracle artifacts replayed",
)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
(output / "manifest.json").write_text(
    json.dumps(
        {str(f.relative_to(output)): sha(f) for f in sorted(output.iterdir()) if f.is_file()},
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(dict(parity_cases=len(parity), quality="PASS", wheel_interfaces="PASS")), flush=True
)
