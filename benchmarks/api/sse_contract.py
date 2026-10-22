"""Socket SSE metadata/finish/UTF-8 audit using the existing qualification server."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qualification"))


def run(source):
    import ten_model_worker as worker

    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(source["path"], offline=True)
    if bundle.model_id != source["repo"] or bundle.revision != source["revision"]:
        raise ValueError("exact checkpoint identity changed")
    rows = []

    def assess(client, route, body, expected=None):
        ordinary = client.post(route, json=body)
        ordinary.raise_for_status()
        payload = ordinary.json()
        choice = payload["choices"][0]
        ordinary_text = choice.get("text", choice.get("message", {}).get("content", ""))
        decoder = codecs.getincrementaldecoder("utf-8")()
        with client.stream("POST", route, json={**body, "stream": True}) as response:
            response.raise_for_status()
            wire = "".join(decoder.decode(chunk) for chunk in response.iter_bytes(chunk_size=1))
            wire += decoder.decode(b"", final=True)
            content_type = response.headers["content-type"]
        events = [
            json.loads(line[6:])
            for line in wire.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        text = "".join(
            event["choices"][0].get("text", event["choices"][0].get("delta", {}).get("content", ""))
            for event in events
            if event.get("choices")
        )
        terminals = [
            event["choices"][0]["finish_reason"]
            for event in events
            if event.get("choices") and event["choices"][0].get("finish_reason")
        ]
        checks = {
            "text_parity": text == ordinary_text and (expected is None or text == expected),
            "finish_parity": terminals == [choice["finish_reason"]],
            "request_identity": len({event.get("id") for event in events}) == 1
            and all(event.get("id") for event in events),
            "model_identity": all(event.get("model") == bundle.model_id for event in events),
            "terminator": wire.endswith("data: [DONE]\n\n") and wire.count("data: [DONE]") == 1,
            "content_type": "text/event-stream" in content_type,
            "no_error": not any("error" in event for event in events),
        }
        usages = [event["usage"] for event in events if event.get("usage")]
        if usages:
            checks["usage_parity"] = usages[-1] == payload["usage"]
        rows.append(
            {
                "route": route,
                "prompt": body.get("prompt"),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL_API",
                "finish_reason": choice["finish_reason"],
                "text": text,
                "one_byte_utf8_decode": True,
                "nonascii_output": any(ord(c) > 127 for c in text),
                "stream_usage_present": bool(usages),
                "event_count": len(events),
            }
        )
        return all(checks.values())

    def audit(client, route, body, expected):
        result = assess(client, route, body, expected)
        if route == "/v1/completions":
            result &= assess(client, route, {**body, "prompt": "café 😀 café 😀", "max_tokens": 8})
        return result

    original = worker.sse_check
    worker.sse_check = audit
    try:
        http = worker.http_checks(
            manager, bundle, source["checks"]["generation"], None, source["chat"]
        )
        return {
            "model_id": bundle.model_id,
            "revision": bundle.revision,
            "status": http["status"],
            "http": http,
            "sse_audits": rows,
            "scope": (
                "Stored exact checkpoint; existing engine on loopback TCP; one-byte client decoding"
            ),
        }
    finally:
        worker.sse_check = original
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = run(json.loads(args.source_result.read_text()))
    report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["source_result_sha256"] = hashlib.sha256(args.source_result.read_bytes()).hexdigest()
    report["production_source_hashes"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src/mlx_one").rglob("*.py")
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
