"""Exact precision selection; unknown, duplicate and empty evidence never passes."""

from __future__ import annotations

import re

DTYPES = {"float32", "float16", "bfloat16"}
LEGACY = re.compile(r"^(dot|rms|attention)-(float32|float16|bfloat16)-(False|True)(?:-|$)")


def operation_dtype(record):
    explicit = record.get("input_dtype")
    legacy = LEGACY.match(record.get("id", ""))
    if explicit is None:
        if legacy is None:
            raise ValueError("missing unambiguous input dtype metadata")
        explicit = legacy.group(2)
    if explicit not in DTYPES:
        raise ValueError("unknown input dtype")
    if legacy and legacy.group(2) != explicit:
        raise ValueError("operation dtype metadata conflicts with legacy ID")
    return explicit


def select_operations(records, dtype, bits):
    if dtype not in DTYPES:
        raise ValueError("unknown requested dtype")
    ids = [r.get("id") for r in records]
    if any(not name for name in ids) or len(ids) != len(set(ids)):
        raise ValueError("missing or duplicate operation IDs")
    selected = [
        r
        for r in records
        if operation_dtype(r) == dtype and (r.get("bits") is None or r.get("bits") == bits)
    ]
    return selected


def operation_verdict(records):
    if not records:
        return "NOT_RUN"
    return "PASS" if all(r.get("status") == "PASS" for r in records) else "FAIL"
