#!/usr/bin/env python3
"""Read-only README evidence/link audit. Run with the repository development venv.

Raw local artifacts remain private; the output contains hashes and aggregate
metrics only. No inference, benchmark rerun, download or source change occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "local/readme-publication/runs/20261104"
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def block(text, start, end):
    assert text.count(start) == text.count(end) == 1, start
    return text[text.index(start) : text.index(end) + len(end)]


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def verify_published_values(md, results, boost):
    profiles = {p["model_type"]: p for p in results["checkpoint_profiles"]}
    admitted = [p for p in profiles.values() if p["python_api"]]
    assert len(admitted) == 4
    registry = read("src/mlx_one/data/native_tasks.json")["profiles"]
    for p in admitted:
        current = next(r for r in registry if r["model_id"] == p["model_id"])
        for key in ["python_api", "cli", "http", "release_qualified", "decision", "revision"]:
            assert current[key] == p[key]
        assert p["decision"] == "EXPERIMENTAL" and not p["release_qualified"]
    q = profiles["distilbert"]["quality"]
    assert f"**{100 * q['accuracy']:.2f}%**" in md and f"({q['correct']}/{q['count']})" in md
    q = profiles["bert_token_classification"]["quality"]
    f1 = (
        2
        * q["true_positive"]
        / (2 * q["true_positive"] + q["false_positive"] + q["false_negative"])
    )
    assert f1 == q["micro_entity_f1"] and f"**{100 * f1:.2f}%**" in md
    q = profiles["clip"]["quality"]
    for key in ["two_label_correct", "ten_label_correct"]:
        assert f"**{q[key]}/{q['images']}**" in md
    for name, title in [
        ("sentiment", "DistilBERT SST-2"),
        ("ner", "BERT NER"),
        ("clip", "CLIP ViT-B/32"),
        ("vad_one_second", "Silero VAD"),
    ]:
        r = results["performance"][name]["requests"]
        assert f"| {title} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} |" in md
    for r, title in zip(
        boost,
        ["Qwen2.5-0.5B-Instruct / Q8", "Qwen3-0.6B / Q8", "SmolLM2-1.7B-Instruct / BF16"],
        strict=True,
    ):
        assert f"| {title} | {r['off']:.2f} | {r['on']:.2f} |" in md
    for title, status in [
        ("ViT-base-patch16-224", "EXPERIMENTAL helper"),
        ("SigLIP2-base-patch16-224 (fixed-resolution SiglipModel)", "BLOCKED"),
        ("DETR ResNet50", "BLOCKED"),
        ("RT-DETR", "NOT IMPLEMENTED"),
        ("Qwen2.5-VL document QA", "BLOCKED"),
        ("Qwen2.5-VL video QA", "BLOCKED"),
        ("Depth Anything V2 Small", "BLOCKED"),
    ]:
        assert f"| {title} | {status} |" in md
    assert "**EXPERIMENTAL**, not release-qualified" in md
    assert "distribution is PROPOSED ONLY" not in md or "standalone" in md


def audit_evidence():
    from benchmarks.audio import readme_generator as audio
    from benchmarks.model_performance import report as text_report
    from benchmarks.ocr import corpus_report as ocr
    from benchmarks.vlm import readme_generator as vlm

    md = (ROOT / "README.md").read_text()
    claims = []
    artifacts = {}

    def artifact(path):
        path = Path(path)
        key = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
        artifacts[key] = {"sha256": sha(ROOT / path), "bytes": (ROOT / path).stat().st_size}
        return key

    def claim(name, path, selector, value, status="HISTORICAL-VERIFIED", **scope):
        claims.append(
            dict(
                claim=name,
                status=status,
                evidence=artifact(path),
                selector=selector,
                value=value,
                **scope,
            )
        )

    inv, rec = text_report.load_records()
    assert block(md, text_report.START, text_report.END) == text_report.render(inv, rec)
    for path in sorted((ROOT / "docs/benchmarks/historical").glob("*.json")):
        d = read(path)
        if "rows" not in d:
            continue
        rows = [
            r for r in d["rows"] if r.get("context") == 128 and r.get("mode") == "production_fp16"
        ]
        # The generator validates and selects each immutable row; retain full
        # metric selectors rather than attributing them to a fresh code run.
        claim(
            "legacy text row " + path.stem,
            path,
            "rows[context=128,mode=production_fp16]",
            rows,
            checkpoint=d.get("checkpoint"),
            revision=d.get("revision"),
            protocol="historical direct-model diagnostic; 16 output-token budget; 3 repeats",
        )
    p = "docs/benchmarks/runs/M01-direct-128-128.json"
    d = read(p)
    claim(
        "standard GPT-2 row",
        p,
        "samples",
        [
            {
                k: v
                for k, v in row.items()
                if k
                in {
                    "ttft_ms",
                    "prefill_tokens_per_second",
                    "decode_tokens_per_second",
                    "peak_inference_mlx_bytes",
                    "generated_tokens",
                    "input_tokens",
                }
            }
            for row in d["samples"]
        ],
        checkpoint=d["checkpoint"],
        revision=d["revision"],
        protocol="128 input / 128 output tokens; standard direct-model protocol",
    )
    plan, records = vlm.records()
    assert vlm.update(md, vlm.render(plan, records)) == md
    for path in sorted((ROOT / "docs/benchmarks/vlm/results").glob("*.json")):
        d = read(path)
        if (
            d.get("workload") != plan.get("readme_workload", "VLM-S1")
            or d.get("status") != "MEASURED"
        ):
            continue
        claim(
            "VLM table row " + path.stem,
            path,
            "metrics",
            d["metrics"],
            checkpoint=d["checkpoint"],
            revision=d["revision"],
            protocol=d["protocol"],
            workload=d["workload"],
            repetitions=d.get("repetitions"),
            max_tokens=d.get("max_tokens"),
        )
    catalog = read("benchmarks/audio/checkpoints.json")
    records = [
        {**read(p), "_evidence_path": p}
        for p in read("docs/benchmarks/audio/publication.json")["records"]
    ]
    assert block(md, audio.START, audio.END) == audio.render(catalog, records).rstrip()
    for d in records:
        claim(
            "audio table row " + d["_evidence_path"].split("/")[-1],
            d["_evidence_path"],
            "metrics,status,protocol,input,decoding",
            {k: d.get(k) for k in ["metrics", "status", "protocol", "input", "decoding"]},
            checkpoint=d["checkpoint"],
            revision=d["revision"],
            protocol=(
                "1 excluded warmup / 5 repeats; task-specific workload; failed rows publish no rate"
            ),
        )
    inventory = read("docs/benchmarks/ocr/inventory.json")["checkpoints"]
    assert (
        block(md, "<!-- MLX-ONE-OCR-BENCHMARK-START -->", "<!-- MLX-ONE-OCR-BENCHMARK-END -->")
        == ocr.render(inventory, write_coverage=False).rstrip()
    )
    for folder in ["omni100-20261025", "hand10-20261025"]:
        for path in sorted((ROOT / "docs/benchmarks/ocr" / folder).glob("O[0-9][0-9].json")):
            d = read(path)
            claim(
                "OCR table row " + folder + "/" + path.stem,
                path,
                "metrics,status",
                {"metrics": d.get("metrics"), "status": d.get("status")},
                checkpoint=d.get("checkpoint"),
                revision=d.get("revision"),
                protocol=(
                    "real corpus; attempted/failed/truncated samples retained; no synthetic rates"
                ),
            )

    # Verify the native sealed evidence byte-for-byte, including historical logs.
    results = read("docs/native_tasks/test-results.json")
    for label, m in results["milestone_manifests"].items():
        assert sha(ROOT / m["path"]) == m["sha256"], label
    sealed = []
    for seal in read("local/native_tasks/completion-index-20261103.json"):
        assert sha(ROOT / seal["manifest"]) == seal["manifest_sha256"], seal["milestone"]
    artifact("local/native_tasks/completion-index-20261103.json")
    for n in range(1, 6):
        path = Path(f"local/native_tasks/runs/m{n}-20261103/manifest.json")
        d = read(path)
        count = 0
        for key, row in d["files"].items():
            file = Path(row["path"]) if "path" in row else path.parent / key
            assert sha(ROOT / file) == row["sha256"], str(file)
            count += 1
        sealed.append(dict(milestone=f"M{n}", path=artifact(path), files_verified=count))
    cp = yaml.safe_load((ROOT / "docs/native_tasks/checkpoints.lock.yaml").read_text())
    folders = [
        "vit",
        "distilbert-sst2",
        "bert-ner",
        "clip-converted",
        "siglip2-fixed",
        "detr",
        None,
        "qwen25-source-metadata",
        "depth-anything",
        "silero-vad",
        "qwen25-existing-q4",
    ]
    checkpoints = []
    for record, folder in zip(cp["checkpoints"], folders, strict=True):
        verified = []
        unavailable = []
        for name, expected in (
            record["asset_hashes"].items() if isinstance(record["asset_hashes"], dict) else []
        ):
            candidates = (
                [
                    ROOT / "local/native_tasks/checkpoints" / folder / name,
                    ROOT / "local/native_tasks/checkpoints" / folder / Path(name).name,
                ]
                if folder
                else []
            )
            actual = next((p for p in candidates if p.is_file()), None)
            if actual:
                assert sha(actual) == expected, str(actual)
                verified.append(name)
            else:
                unavailable.append(name)
        checkpoints.append(
            dict(
                model_id=record["model_id"],
                revision=record["revision"],
                asset_hashes=record["asset_hashes"],
                verified_assets=verified,
                unavailable_assets=unavailable,
            )
        )
    sources = yaml.safe_load((ROOT / "docs/native_tasks/sources.lock.yaml").read_text())
    verified_source = 0
    for name, expected in sources["transformers"]["files"].items():
        data = git(
            "show", sources["transformers"]["commit"] + ":" + name, cwd=ROOT.parent / "transformers"
        )
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
        verified_source += 1
    artifact("docs/native_tasks/checkpoints.lock.yaml")
    artifact("docs/native_tasks/sources.lock.yaml")
    for profile in results["checkpoint_profiles"]:
        lock = next(p for p in checkpoints if p["model_id"] == profile["model_id"])
        assert lock["revision"] == profile["revision"]
        claim(
            "task decision " + profile["model_id"],
            "docs/native_tasks/test-results.json",
            "checkpoint_profiles[model_id=" + profile["model_id"] + "]",
            profile,
        )
    # Recompute the published quality aggregates from sealed evaluation records.
    qnative = read("local/native_tasks/runs/m1-20261103/quality-native.json")
    qreference = read("local/native_tasks/runs/m1-20261103/quality-hf.json")
    correct = sum(r["prediction"] == r["label"] for r in qnative["rows"])
    assert len(qnative["rows"]) == 872 and correct == 794
    assert all(
        a["prediction"] == b["prediction"]
        for a, b in zip(qnative["rows"], qreference["rows"], strict=True)
    )
    artifact("local/native_tasks/runs/m1-20261103/quality-hf.json")
    claim(
        "SST-2 quality aggregate",
        "local/native_tasks/runs/m1-20261103/quality-native.json",
        "rows[*]: sum(prediction == label), length; paired HF prediction equality",
        dict(count=872, correct=correct, accuracy=correct / 872, source_agreement=872),
    )
    ner = read("local/native_tasks/runs/m2-20261103/quality-summary.json")
    assert ner["native"] == ner["hf"]
    assert ner["native"]["micro_entity_f1"] == 2 * 153 / (2 * 153 + 156 + 45)
    claim(
        "NER quality aggregate",
        "local/native_tasks/runs/m2-20261103/quality-summary.json",
        "native,hf: exact strict-span summary equality",
        ner,
    )
    clip = read("local/native_tasks/runs/m2-20261103/clip/matrix-summary.json")
    clip_metrics = {
        str(n): dict(
            count=sum(r["candidates"] == n for r in clip),
            correct=sum(r["candidates"] == n and r["correct"] for r in clip),
            full_trace_pass=sum(r["candidates"] == n and r["source_parity_pass"] for r in clip),
        )
        for n in [2, 10, 100]
    }
    assert clip_metrics["2"]["correct"] == 6 and clip_metrics["10"]["correct"] == 5
    assert clip_metrics["100"]["full_trace_pass"] == 0
    assert all(r["top1_agreement"] for r in clip) and len(clip) == 18
    claim(
        "CLIP quality and rejected profile",
        "local/native_tasks/runs/m2-20261103/clip/matrix-summary.json",
        "group by candidates; sum(correct),sum(source_parity_pass); all(top1_agreement)",
        clip_metrics,
    )
    vad = read("local/native_tasks/runs/m3-20261103/quality/native.json")
    assert [vad[k] for k in ["tp", "fp", "fn", "tn"]] == [10, 0, 0, 3]
    claim(
        "VAD clip-presence quality",
        "local/native_tasks/runs/m3-20261103/quality/native.json",
        "metric,tp,fp,fn,tn",
        {k: vad[k] for k in ["metric", "tp", "fp", "fn", "tn"]},
    )
    full = read("local/native_tasks/runs/m3-20261103/silero-full-comparison.json")
    assert full["passed"]
    claim(
        "VAD full state/probability parity",
        "local/native_tasks/runs/m3-20261103/silero-full-comparison.json",
        "passed,tensors: all 1875 real-audio chunks",
        full,
    )
    for name, perf in results["performance"].items():
        assert sha(ROOT / perf["raw"]) == perf["sha256"]
        d = read(perf["raw"])
        metric = d["python_task"]
        values = metric["values_ms"]
        assert len(values) == metric["samples"] == 20 and d["warmups"] == 1
        assert statistics.median(values) == metric["median_ms"]
        assert sorted(values)[math.ceil(0.95 * len(values)) - 1] == metric["p95_ms"]
        claim(
            "task latency " + name,
            perf["raw"],
            "python_task",
            metric,
            protocol=(
                "20 timed calls / 1 excluded warmup / 1 process; preparation plus "
                "synchronized native execution"
            ),
        )
    for name in ["default", "native_selection"]:
        d = results["tests"][name]
        log = (ROOT / d["log"]).read_text()
        assert f"{d['passed']} passed" in log
        if name == "default":
            assert "79 skipped" in log and "9 subtests passed" in log
        claim(
            "historical " + name + " suite",
            d["log"],
            "pytest terminal summary",
            d,
            protocol="historical 2026-11-03; suites overlap",
        )
    assert sha(ROOT / results["wheel"]["path"]) == results["wheel"]["sha256"]
    claim("native wheel", "docs/native_tasks/test-results.json", "wheel", results["wheel"])
    # Full-response HTTP request rates, preserving each cohort and all outputs.
    boost = []
    for model, off, on in [
        ("qwen2", "expanded-qwen2-off", "expanded-qwen2-on"),
        ("qwen3", "expanded-qwen3-off", "expanded-qwen3-on"),
        ("smollm2", "original-protocol-off", "original-protocol-fast"),
    ]:
        pair = {}
        for flag, folder in [("off", off), ("on", on)]:
            path = f"local/mlxboost/runs/confirmation-20261101/{folder}/per_request.csv"
            with (ROOT / path).open() as f:
                rows = [
                    r
                    for r in csv.DictReader(f)
                    if r["model"] == model and r["backend"] == "mlx-one"
                ]
            assert len(rows) == (6 if model == "smollm2" else 50)
            for row in rows:
                assert math.isclose(
                    float(row["output_tokens_per_second"]),
                    int(row["completion_tokens"]) / float(row["latency_seconds"]),
                    rel_tol=1e-12,
                )
            manifest_path = f"local/mlxboost/runs/confirmation-20261101/{folder}/manifest.json"
            manifest = read(manifest_path)
            checkpoint = next(c for c in manifest["models"] if c["key"] == model)
            assets = []
            unavailable = []
            for name, expected in checkpoint["files_sha256"].items():
                file = Path(checkpoint["path"]) / name
                if file.is_file():
                    assert sha(file) == expected, "MLXBoost checkpoint asset " + name
                    assets.append(name)
                else:
                    unavailable.append(name)
            artifact(manifest_path)
            value = statistics.median(float(r["output_tokens_per_second"]) for r in rows)
            claim(
                "MLXBoost " + model + " " + flag,
                path,
                "model=" + model + ",backend=mlx-one; median(output_tokens_per_second)",
                dict(
                    median_tok_s=value,
                    timed_requests=len(rows),
                    prompt_hashes=[r["prompt_sha256"] for r in rows],
                ),
                protocol="full-response HTTP SSE; 1 excluded warmup per process; no EOS filtering",
                checkpoint=checkpoint["repo"],
                revision=checkpoint["revision"],
                files_sha256=checkpoint["files_sha256"],
                assets_verified_now=assets,
                unavailable_assets=unavailable,
                input_tokens=manifest["protocol"]["prompt_tokens"],
                output_budget=manifest["protocol"]["output_budget"],
            )
            pair[flag] = value
        boost.append(dict(model=model, **pair))
    assert [[round(r[k], 2) for k in ["off", "on"]] for r in boost] == [
        [22.94, 109.17],
        [21.42, 96.42],
        [15.39, 24.56],
    ]
    verify_published_values(md, results, boost)
    claim(
        "MLXBoost Qwen paired outputs",
        "docs/mlxboost/performance-summary.json",
        "paired-qwen2,paired-qwen3",
        {
            k: read("docs/mlxboost/performance-summary.json")[k]
            for k in ["paired-qwen2", "paired-qwen3"]
        },
    )
    # Report-backed bounded results; do not assert missing historical raw traces.
    gpt = read("docs/validation/gpt-oss-kilo-2026-10-23.json")["hardware"]["checks"]["stream_tools"]
    gpt_fields = {
        k: statistics.median(r["native_metrics"][k] for r in gpt)
        for k in ["prompt_tokens_per_second", "decode_tokens_per_second", "time_to_first_token_ms"]
    }
    gpt_fields["peak_mlx_GiB"] = max(r["native_metrics"]["memory_peak_bytes"] for r in gpt) / 2**30
    gpt_fields["first_tool_delta_s"] = statistics.median(r["first_tool_delta_s"] for r in gpt)
    assert [round(v, 2) for v in list(gpt_fields.values())[:3]] == [75.95, 18.96, 2355.38]
    assert round(gpt_fields["peak_mlx_GiB"], 3) == 10.608
    assert round(gpt_fields["first_tool_delta_s"], 3) == 7.127
    claim(
        "GPT-OSS HTTP / Kilo",
        "docs/validation/gpt-oss-kilo-2026-10-23.json",
        (
            "hardware.checks.stream_tools[*].native_metrics; median except "
            "max(memory_peak_bytes)/2**30; median(first_tool_delta_s)"
        ),
        gpt_fields,
        provenance_limit=(
            "historical report; checkpoint was local, not a newly resolved immutable weight lock"
        ),
    )
    claim(
        "SAM3 single-truck fixture",
        "docs/sam3-qualification-m4-32gb.md",
        "streaming/offline comparison tables",
        {
            "minimum_mask_iou_rounded": 0.99998,
            "maximum_score_difference": 0.00007511,
            "box_difference": 0,
            "object_id": 0,
            "peak_GB_decimal_approx": 8.41,
        },
        provenance_limit=(
            "pinned checkpoint/reference report; controlled fixtures only; raw"
            " archived trace availability not certified by this audit"
        ),
    )
    # Current semantic evidence is code, not a new numerical benchmark.
    for path in [
        "src/mlx_one/mlxboost/config.py",
        "src/mlx_one/engine/modes.py",
        "src/mlx_one/engine/construction.py",
        "src/mlx_one/engine/pressure.py",
        "src/mlx_one/data/native_tasks.json",
    ]:
        artifact(path)
    qualification_claims = []
    heading = ""
    native_section = False
    boost_section = False
    for lineno, line in enumerate(md.splitlines(), 1):
        if "<!-- mlx-one:verified-task-results:start -->" in line:
            native_section = True
        elif "<!-- mlx-one:verified-task-results:end -->" in line:
            native_section = False
        if "<!-- mlx-one:mlxboost-summary:start -->" in line:
            boost_section = True
        elif "<!-- mlx-one:mlxboost-summary:end -->" in line:
            boost_section = False
        if line.startswith("##"):
            heading = line.lstrip("# ")
        if not re.search(
            r"qualif|validat|parity|gates pass|anchors pass|fixture.*pass|smoke.*pass", line, re.I
        ):
            continue
        evidence = ["docs/model-admission.md"]
        if "audio" in heading.lower() or any(
            n in line.lower()
            for n in ["moonshine", "parakeet", "qwen3_asr", "forced_aligner", "whisper"]
        ):
            evidence = ["docs/audio-qualification-m4-32gb.md"]
        elif "segmen" in heading.lower() or "sam3" in line.lower() or "mask-iou" in line.lower():
            evidence = ["docs/sam3-qualification-m4-32gb.md"]
        elif "mlxboost" in heading.lower():
            evidence = ["src/mlx_one/mlxboost/config.py", "docs/mlxboost/qualification-matrix.md"]
        elif any(
            n in heading.lower() for n in ["candidate", "experimental native", "sentiment example"]
        ):
            evidence = ["docs/native_tasks/test-results.json"]
        elif "performance" in heading.lower():
            evidence = ["docs/benchmarks/inventory.json"]
        if native_section:
            evidence = ["docs/native_tasks/test-results.json", "src/mlx_one/data/native_tasks.json"]
        elif boost_section:
            evidence = ["src/mlx_one/mlxboost/config.py", "docs/mlxboost/qualification-matrix.md"]
        links = re.findall(r"\]\(([^)#]+)", line)
        evidence += [
            link for link in links if not link.startswith("http") and (ROOT / link).is_file()
        ]
        for file in evidence:
            artifact(file)
        qualification_claims.append(
            dict(
                readme_line=lineno,
                status="VERIFIED"
                if any(file.startswith("src/") for file in evidence)
                or evidence == ["docs/model-admission.md"]
                else "HISTORICAL-VERIFIED",
                text=line,
                evidence=list(dict.fromkeys(evidence)),
                review=(
                    "Scope retained; no qualification promotion. See claim-group "
                    "decision in audit.md."
                ),
            )
        )
    baseline = read("local/readme-publication/runs/20261104/baseline.json")
    for path, expected in baseline["non_documentation_files"].items():
        assert sha(ROOT / path) == expected, "non-documentation changed: " + path
    for path, expected in baseline["protected_files"].items():
        assert sha(ROOT / path) == expected, "protected file changed: " + path
    assert sha(ROOT / ".git/index") == baseline["git_index_sha256"], "main worktree index changed"
    legacy = block(md, text_report.START, text_report.END)
    assert (
        hashlib.sha256(legacy.encode()).hexdigest() == baseline["legacy_model_performance_sha256"]
    )
    return dict(
        schema_version=1,
        audit_date="2026-11-04",
        base_head=baseline["head"],
        scope=(
            "documentation verification; performance and qualification evidence remains historical"
        ),
        artifacts=artifacts,
        claims=claims,
        qualification_claims=qualification_claims,
        checkpoint_locks=checkpoints,
        source_commits=results["sources"],
        transformers_source_files_verified=verified_source,
        milestone_seals=sealed,
        preservation=dict(
            non_documentation_files_unchanged=len(baseline["non_documentation_files"]),
            protected_files_unchanged=len(baseline["protected_files"]),
            model_performance_sha256=baseline["legacy_model_performance_sha256"],
            main_index_unchanged=True,
        ),
        omissions=[
            "No QUALIFIED claim for new tasks",
            "No general tracking/depth/VAD boundary-quality claim",
            "No model-forward speedup claim from HTTP rates",
            "No standalone MLXBoost package or installation command",
            "No new performance measurement",
            (
                "Historical raw SAM3 archives not certified; GPT-OSS immutable "
                "checkpoint lock unavailable"
            ),
        ],
    )


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if d.get("id"):
            self.ids.append(d["id"])
        if tag == "a" and d.get("name"):
            self.ids.append(d["name"])
        for key in ["href"] if tag == "a" else ["src"] if tag == "img" else []:
            if d.get(key):
                self.links.append(d[key])


def parse_md(path):
    text = path.read_text()
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = parser.parse(text)
    parsed = HTMLLinks()
    parsed.feed(parser.render(text))
    seen = {}
    for i, t in enumerate(tokens):
        if t.type == "heading_open":
            inline = tokens[i + 1]
            title = "".join(
                x.content for x in inline.children or [] if x.type in {"text", "code_inline"}
            )
            slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            parsed.ids.append(slug + ("-" + str(n) if n else ""))
    for token in tokens:
        if token.type == "table_open":
            rows = text.splitlines()[token.map[0] : token.map[1]]
            widths = [len(re.split(r"(?<!\\)\|", row.strip().strip("|"))) for row in rows]
            assert len(set(widths)) == 1, "inconsistent table columns: " + str(
                path.relative_to(ROOT)
            )
    # CommonMark tolerates open fences. Publication does not.
    fence = None
    for line in text.splitlines():
        m = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if m:
            value = m[1]
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
    assert fence is None, "unclosed fence: " + str(path.relative_to(ROOT))
    assert len(parsed.ids) == len(set(parsed.ids)), "duplicate anchor: " + str(
        path.relative_to(ROOT)
    )
    return text, parser.render(text), parsed


def audit_markdown():
    files = [
        ROOT / "README.md",
        ROOT / "docs/architecture/mlxboost-module-boundary.md",
        OUT / "audit.md",
    ]
    files += sorted(OUT.glob("*.md"))
    files = list(dict.fromkeys(p for p in files if p.exists()))
    errors = []
    external = set()
    local = 0
    for path in files:
        text, html, parsed = parse_md(path)
        assert not re.search(r"/[U]sers/|github_pat_|ghp_[A-Za-z0-9]|git@github\.com", text), (
            "private path/credential/SSH URL"
        )
        for url in parsed.links:
            parts = urlsplit(url)
            if parts.scheme in {"http", "https"}:
                external.add(url)
                continue
            if parts.scheme or not parts.path and not parts.fragment:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.exists():
                errors.append(
                    dict(file=str(path.relative_to(ROOT)), url=url, error="missing local target")
                )
                continue
            local += 1
            if parts.fragment and target.suffix == ".md":
                anchors = parse_md(target)[2].ids
                if unquote(parts.fragment) not in anchors:
                    errors.append(
                        dict(file=str(path.relative_to(ROOT)), url=url, error="missing anchor")
                    )
        if path.name == "README.md":
            for name in ["verified-task-results", "mlxboost-summary"]:
                block(text, f"<!-- mlx-one:{name}:start -->", f"<!-- mlx-one:{name}:end -->")
            preview = (
                (
                    '<!doctype html><html lang="en"><meta charset="utf-8"><base '
                    'href="../../../"><title>MLX-One README '
                    "preview</title><style>body{max-width:1120px;margin:2rem "
                    "auto;padding:0 1rem;font:16px/1.5 "
                    "system-ui}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow:auto}th,td{border:1px"
                    " solid #ddd;padding:6px "
                    "10px}pre{overflow:auto;background:#f6f8fa;padding:1rem}a{color:#0969da}</style>"
                )
                + html
                + "</html>\n"
            )
            (OUT / "README.preview.html").write_text(preview)
    result = dict(
        files=len(files), local_links_checked=local, errors=errors, external_urls=sorted(external)
    )
    (RUN / "markdown-check.json").write_text(json.dumps(result, indent=2) + "\n")
    assert not errors, json.dumps(errors)
    return {k: v for k, v in result.items() if k != "external_urls"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        md = (ROOT / "README.md").read_text()
        results = read("docs/native_tasks/test-results.json")
        claims = read(OUT / "evidence-map.json")["claims"]
        boost = []
        for model in ["qwen2", "qwen3", "smollm2"]:
            boost.append(
                {
                    flag: next(
                        c["value"]["median_tok_s"]
                        for c in claims
                        if c["claim"] == "MLXBoost " + model + " " + flag
                    )
                    for flag in ["off", "on"]
                }
            )
        verify_published_values(md, results, boost)
        for old, new in [
            ("91.06%", "92.06%"),
            ("14.95", "0.01"),
            ("109.17", "9.17"),
            ("| DETR ResNet50 | BLOCKED |", "| DETR ResNet50 | QUALIFIED |"),
        ]:
            assert old in md
            try:
                verify_published_values(md.replace(old, new), results, boost)
            except AssertionError:
                continue
            raise AssertionError("checker accepted mutated claim: " + old)
        print("PASS: rejects altered accuracy, latency, HTTP rate and qualification status")
        return
    if not args.markdown_only:
        report = audit_evidence()
        data = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        assert not re.search(r"/[U]sers/", data)
        (OUT / "evidence-map.json").write_text(data)
        print(
            json.dumps(
                {
                    "claims": len(report["claims"]),
                    "seals": report["milestone_seals"],
                    "preservation": report["preservation"],
                }
            )
        )
    print(json.dumps(audit_markdown()))


if __name__ == "__main__":
    main()
