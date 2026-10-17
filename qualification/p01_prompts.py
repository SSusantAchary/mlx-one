"""Versioned offline rendered-token fixtures; loading never imports MLX."""
from __future__ import annotations

import json
import re
from pathlib import Path

from mlx_one.p0_engine_qualification import digest
from qualification.p01_math import canonical_digest

VERSION = "native-math-prompts-v1"


def load_prompts(path, metadata):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (data.get("schema_version") != VERSION
            or data.get("checkpoint") != metadata["model_id"]
            or data.get("revision") != metadata["revision"]
            or not re.fullmatch(r"[0-9a-f]{40}", data.get("revision", ""))
            or data.get("assets_sha256") != canonical_digest(metadata["assets"])
            or not data.get("dataset_identity") or not data.get("provenance")):
        raise ValueError("prompt fixture identity, provenance or immutable assets mismatch")
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty rendered-token fixtures required")
    seen, widths = set(), set()
    for record in records:
        tokens, text = record.get("token_ids"), record.get("text")
        if (not isinstance(tokens, list) or not tokens or not isinstance(text, str) or not text
                or any(type(token) is not int or token < 0 for token in tokens)
                or not record.get("id") or record["id"] in seen or len(tokens) in widths
                or record.get("token_sha256") != canonical_digest(tokens)):
            raise ValueError("invalid, duplicate or unchecked rendered-token fixture")
        seen.add(record["id"])
        widths.add(len(tokens))
    return data


def resolve_prompt(path, checksum, metadata, tokenizer, width):
    if digest(path) != checksum:
        raise ValueError("frozen prompt manifest changed")
    data = load_prompts(path, metadata)
    matching = [record for record in data["records"] if len(record["token_ids"]) == width]
    if len(matching) != 1:
        raise ValueError("exact declared rendered-token fixture missing")
    record = matching[0]
    if list(tokenizer.encode(record["text"])) != record["token_ids"]:
        raise ValueError("actual checkpoint tokenization differs from frozen fixture")
    return record["text"], list(record["token_ids"])
