"""OCR accuracy metrics and fail-closed measurement validation."""

import math
import re
import statistics
import unicodedata


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_metrics(reference, hypothesis):
    # Empty references need an explicit scoring policy; do not invent denominators.
    def score(r, h):
        words = r.split()
        return {
            "cer": distance(r, h) / len(r) if r else None,
            "wer": distance(words, h.split()) / len(words) if words else None,
            "exact_match": r == h,
        }

    return {
        "raw": score(reference, hypothesis),
        "normalized": score(normalize(reference), normalize(hypothesis)),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(record):
    require(
        record.get("schema_version") == "mlx-one-ocr-benchmark-v1", "unknown OCR benchmark schema"
    )
    require(
        record.get("status") in {"NOT_RUN", "MEASURED", "FAILED"}, "invalid OCR benchmark status"
    )
    require(bool(record.get("checkpoint")), "checkpoint identity missing")
    if record["status"] != "MEASURED":
        return
    require(
        not record.get("mock") and record.get("fixture_kind") in {"synthetic", "real"},
        "mock or unidentified fixtures cannot be published",
    )
    require(
        bool(re.fullmatch("[0-9a-f]{40,64}", str(record.get("revision", "")))),
        "immutable checkpoint revision required",
    )
    require(
        record.get("measurement_type") in {"direct-model", "python-engine", "cli", "http"},
        "measurement type required",
    )
    require(
        record.get("hardware", {}).get("chip") and record["hardware"].get("memory_bytes", 0) > 0,
        "hardware evidence missing",
    )
    require(record.get("software", {}).get("mlx"), "MLX version missing")
    for name in ["implementation_sha256", "runner_sha256", "workload_sha256", "checkpoint_sha256"]:
        require(bool(re.fullmatch("[0-9a-f]{64}", str(record.get(name, "")))), f"{name} required")
    require(
        record.get("protocol", {}).get("batch_size") == 1
        and record["protocol"].get("warmups") == 1
        and record["protocol"].get("repetitions") == 5
        and record["protocol"].get("statistic") == "median",
        "invalid OCR protocol",
    )
    samples = record.get("samples", [])
    require(len(samples) == 5, "five actual samples required")
    require("reference" in record, "ground-truth transcript required")
    for sample in samples:
        require(
            sample.get("accuracy")
            == text_metrics(record["reference"], sample.get("transcript", "")),
            "accuracy must be recomputed from actual transcript and reference",
        )
        for key in ["latency_seconds", "peak_mlx_memory_gib", "peak_process_memory_gib"]:
            value = sample.get(key)
            require(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                f"invalid {key}",
            )
        require(sample["latency_seconds"] > 0 and sample.get("pages", 0) > 0, "no completed pages")
        require(sample.get("output_tokens", -1) >= 0, "actual token count missing")
    for key in ["latency_seconds", "peak_mlx_memory_gib"]:
        require(
            record.get("metrics", {}).get(key) == statistics.median(s[key] for s in samples),
            "summary must match actual sample medians",
        )
    throughput = statistics.median(s["pages"] / s["latency_seconds"] for s in samples)
    require(record["metrics"].get("pages_per_second") == throughput, "invalid page throughput")
    require(
        record["metrics"].get("cer")
        == statistics.median(s["accuracy"]["normalized"]["cer"] for s in samples),
        "CER must match measured transcripts",
    )


def box_iou(left, right):
    """Pixel xyxy IoU. Degenerate or reversed boxes are invalid annotations."""
    for box in (left, right):
        require(
            len(box) == 4
            and all(math.isfinite(v) for v in box)
            and box[2] > box[0]
            and box[3] > box[1],
            "invalid annotated/predicted box",
        )
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union


def region_metrics(reference, predicted, threshold=0.5):
    """Greedy descending-IoU matching, scoped to page; not COCO AP or polygon IoU."""
    candidates = []
    for i, actual in enumerate(predicted):
        for j, target in enumerate(reference):
            if actual["page"] == target["page"]:
                score = box_iou(actual["bbox"], target["bbox"])
                if score >= threshold:
                    candidates.append((score, i, j))
    used_actual, used_target = set(), set()
    matches = []
    for score, i, j in sorted(candidates, reverse=True):
        if i in used_actual or j in used_target:
            continue
        used_actual.add(i)
        used_target.add(j)
        matches.append((i, j, score))
    precision = len(matches) / len(predicted) if predicted else 0
    recall = len(matches) / len(reference) if reference else 0
    errors = [
        sum(abs(a - b) for a, b in zip(predicted[i]["bbox"], reference[j]["bbox"], strict=True)) / 4
        for i, j, _ in matches
    ]
    pairs = []
    for index, (i, j, _) in enumerate(matches):
        for other, other_target, _ in matches[index + 1 :]:
            if reference[j]["page"] == reference[other_target]["page"] and all(
                "reading_order" in row and row["reading_order"] is not None
                for row in [predicted[i], predicted[other], reference[j], reference[other_target]]
            ):
                pairs.append(
                    (predicted[i]["reading_order"] < predicted[other]["reading_order"])
                    == (reference[j]["reading_order"] < reference[other_target]["reading_order"])
                )
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "mean_matched_iou": statistics.mean(s for _, _, s in matches) if matches else None,
        "mean_matched_coordinate_error_pixels": statistics.mean(errors) if errors else None,
        "matched_reading_order_pair_accuracy": statistics.mean(pairs) if pairs else None,
        "evaluated_order_pairs": len(pairs),
        "matched_regions": len(matches),
        "reference_regions": len(reference),
        "predicted_regions": len(predicted),
    }


def table_metrics(reference, predicted):
    """Position-aligned cell/span fidelity; not TEDS or semantic table equivalence."""
    expected = [cell for row in reference["rows"] for cell in row]
    actual = [cell for row in predicted.get("rows", []) for cell in row]
    text_matches = 0
    span_matches = 0
    for i, cell in enumerate(expected):
        other = actual[i] if i < len(actual) else {}
        text_matches += normalize(cell.get("text", "")) == normalize(other.get("text", ""))
        span_matches += (
            all(cell.get(key, 1) == other.get(key, 1) for key in ("rowspan", "colspan"))
            if other
            else 0
        )
    return {
        "row_cell_counts_exact": [len(r) for r in reference["rows"]]
        == [len(r) for r in predicted.get("rows", [])],
        "cell_text_exact_fraction": text_matches / len(expected) if expected else None,
        "cell_span_exact_fraction": span_matches / len(expected) if expected else None,
        "reference_cells": len(expected),
        "predicted_cells": len(actual),
    }


def auxiliary_metrics(fixture, result, *, geometry=False, structured_table=False, formula=False):
    quality = {"regions": None, "tables": None, "formula_normalized_exact": None}
    if geometry and "reference_regions" in fixture:
        predicted = [
            {"page": p.page, "bbox": element.bbox, "reading_order": element.reading_order}
            for p in result.pages
            for element in p.elements
            if element.bbox is not None
        ]
        quality["regions"] = region_metrics(fixture["reference_regions"], predicted)
    if structured_table and "reference_tables" in fixture:
        predicted = [table for p in result.pages for table in p.tables]
        quality["tables"] = [
            table_metrics(table, predicted[i] if i < len(predicted) else {})
            for i, table in enumerate(fixture["reference_tables"])
        ]
    if formula and "reference_latex" in fixture:
        predicted = [formula for p in result.pages for formula in p.formulas]

        def compact(value):
            return re.sub(r"\s+", "", value)

        quality["formula_normalized_exact"] = [
            compact(target) == compact(predicted[i]) if i < len(predicted) else False
            for i, target in enumerate(fixture["reference_latex"])
        ]
    return quality
