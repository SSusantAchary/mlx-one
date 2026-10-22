"""Frozen, bounded task quality, independent of tensor and runtime gates."""

import hashlib
import json
import re


def classify(case, text, finish_reason):
    """Keep expected answers fixed; distinguish a failed truncated response."""
    passed = bool(text.strip())
    if case["kind"] == "exact":
        passed = text.strip() == case["expected"]
    elif case["kind"] == "json":
        try:
            passed = json.loads(text) == {"name": "Lin", "count": 2}
        except ValueError:
            passed = False
    if passed:
        return "PASS"
    if finish_reason == "length":
        return "TRUNCATED"
    if case["id"] == "arithmetic" and re.fullmatch(
        r"17\s*\+\s*25\s*=\s*" + re.escape(case["expected"]), text.strip()
    ):
        # Correct arithmetic with extra text still fails the exact-format task.
        return "FAIL_INSTRUCTION"
    if case["id"] == "bounded-retrieval" and re.fullmatch(
        r'The secret value is ["*]*' + re.escape(case["expected"]) + r'["*]*[.!]?', text.strip()
    ):
        return "FAIL_INSTRUCTION"
    return {
        "arithmetic": "FAIL_ARITHMETIC",
        "json": "FAIL_JSON",
        "bounded-retrieval": "FAIL_RETRIEVAL",
        "repetition": "FAIL_REPETITION",
    }.get(case["id"], "FAIL_INSTRUCTION")


def evaluate(bundle, engine, fixture_path, *, chat_applicable):
    from mlx_one.text import TextGenerationOptions

    fixture = json.loads(fixture_path.read_text())
    rows = []
    for case in fixture["cases"]:
        if case["instruction"] and (
            not chat_applicable or not bundle.chat_template or not bundle.chat_template.template
        ):
            rows.append(
                {
                    "id": case["id"],
                    "status": (
                        "UNSUPPORTED_TASK"
                        if fixture["schema_version"].endswith("v2")
                        else "UNSUPPORTED"
                    ),
                    "reason": "instruction/chat task has no applicable template contract",
                }
            )
            continue
        prompt = case.get("prompt", "").replace("{filler}", "background " * 128)
        messages = case.get("messages")
        template_options = {}
        if case["instruction"]:
            if "enable_thinking" in bundle.chat_template.template:
                template_options["enable_thinking"] = False
            prompt = bundle.chat_template.render(
                messages or [{"role": "user", "content": prompt}], **template_options
            )
        chunks = list(
            engine.stream(
                prompt,
                options=TextGenerationOptions(
                    max_tokens=case.get("max_tokens", fixture["max_tokens"]), temperature=0
                ),
            )
        )
        text = "".join(c.text for c in chunks)
        status = classify(case, text, chunks[-1].finish_reason)
        if not fixture["schema_version"].endswith("v2") and status != "PASS":
            status = "FAIL_BEHAVIOR"
        words = text.split()
        rows.append(
            {
                "id": case["id"],
                "status": status,
                "text": text,
                "token_ids": [c.token_id for c in chunks if c.token_id is not None],
                "prompt_tokens": len(bundle.tokenizer.encode(prompt)),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_token_ids": bundle.tokenizer.encode(prompt),
                "max_tokens": case.get("max_tokens", fixture["max_tokens"]),
                "template_options": template_options,
                "finish_reason": chunks[-1].finish_reason,
                "unique_word_ratio": len(set(words)) / len(words) if words else None,
            }
        )
    scored = [r for r in rows if r["status"] not in {"UNSUPPORTED", "UNSUPPORTED_TASK"}]
    return {
        "schema_version": fixture["schema_version"],
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "status": "PASS" if all(r["status"] == "PASS" for r in scored) else "FAIL_BEHAVIOR",
        "passed": sum(r["status"] == "PASS" for r in scored),
        "scored": len(scored),
        "cases": rows,
        "metric_contract": fixture["metrics"],
    }
