"""Versioned proposed adjudication: independent dimensions and explicit missing evidence."""

from __future__ import annotations

VERSION = "numerical-contract-v2-proposal"


def frozen_reference_verdict(rows, *, complete_scope=True):
    if not complete_scope:
        return "NOT_ESTABLISHED"
    return evidence_status(rows)


def require_profile_alignment(left, right):
    fields = ("checkpoint_revision", "weight_profile", "kv_policy", "input_ids_sha256")
    if any(left.get(f) is None or right.get(f) is None for f in fields):
        raise ValueError("missing comparison profile identity")
    if any(left[f] != right[f] for f in fields):
        raise ValueError("comparison profile or state policy mismatch")


def evidence_status(rows):
    if not rows:
        return "NOT_RUN"
    states = [r.get("status", "NOT_RUN") for r in rows]
    if "FAIL" in states:
        return "FAIL"
    if any(s != "PASS" for s in states):
        return "NOT_ESTABLISHED"
    return "PASS"


def evaluate(dimensions, *, reliability_disposition="PENDING", approval="PENDING"):
    required = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
    if set(dimensions) != required:
        raise ValueError("all six qualification dimensions required")
    accuracy = [dimensions[k] for k in ["Q1", "Q2", "Q3", "Q5", "Q6"]]
    if "FAIL" in accuracy:
        outcome = "FAIL_MANDATORY_EVIDENCE"
    elif any(s != "PASS" for s in accuracy):
        outcome = "BLOCKED_INSUFFICIENT_EVIDENCE"
    elif dimensions["Q4"] != "PASS" and reliability_disposition != "APPROVED":
        outcome = "PENDING_NUMERICAL_DISPOSITION"
    else:
        outcome = "PROVISIONAL_NUMERICAL_PASS"
    return {
        "version": VERSION,
        "dimensions": dimensions,
        "outcome": outcome,
        "reliability_disposition": reliability_disposition,
        "contract_approval": approval,
        "alpha_supported": False,
        "historical_frozen_overridden": False,
    }
