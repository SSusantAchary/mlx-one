#!/usr/bin/env python3
"""Small authored-label document QA diagnostic; full source model gate is separate."""

import argparse
import json
import re
import resource
import time
from collections import Counter
from pathlib import Path

from mlx_one.vision.document_qa import DocumentQAEngine


def words(text):
    return re.findall(r"\w+", text.casefold())


def score(actual, expected):
    a, b = words(actual), words(expected)
    overlap = sum((Counter(a) & Counter(b)).values())
    return {"exact_match": a == b, "token_f1": 2 * overlap / (len(a) + len(b)) if a or b else 1.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("cases", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    cases = json.loads(args.cases.read_text())
    rows = []
    with DocumentQAEngine(args.checkpoint) as engine:
        for row in cases["cases"]:
            start = time.perf_counter()
            result = engine.ask(row["image"], row["question"])
            scores = [score(result.answer, a) for a in row["answers"]]
            rows.append(
                {
                    "id": row["id"],
                    "result": result.to_dict(),
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "exact_match": any(s["exact_match"] for s in scores),
                    "token_f1": max(s["token_f1"] for s in scores),
                }
            )
        import mlx.core as mx

        peak = mx.get_peak_memory()
    report = {
        "count": len(rows),
        "exact_match": sum(r["exact_match"] for r in rows) / len(rows),
        "mean_token_f1": sum(r["token_f1"] for r in rows) / len(rows),
        "rows": rows,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_mlx_bytes": peak,
        "limitations": (
            "Two real pages, six authored questions; "
            "no source decoder QA comparison after vision gate failed"
        ),
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
