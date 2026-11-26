"""Read-only exact qualification. No compilation, tuning or model registry mutation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path

from .schemas import KERNEL_ID, SCHEMA_VERSION, THREADGROUPS, KernelCandidate

MAX_RECORD_BYTES = 4 * 1024 * 1024


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_json(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_json(path):
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("record size limit")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate manifest field")
            result[key] = value
        return result

    return json.loads(
        data,
        object_pairs_hook=unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
    )


def write_immutable(directory, record):
    """Atomically publish a content-addressed record; never overwrite another run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(record)
    target = directory / (hashlib.sha256(payload).hexdigest() + ".json")
    fd, name = tempfile.mkstemp(prefix=".mlxboost-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(name, target)  # Atomic create-if-absent, unlike replace().
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != payload:
                raise ValueError("immutable evidence collision") from None
    finally:
        Path(name).unlink(missing_ok=True)
    return target


class Registry:
    def __init__(self, manifest=None):
        self.manifest = Path(manifest) if manifest is not None else None

    def entries(self):
        if self.manifest is None:
            return [], "missing_evidence"
        try:
            data = read_json(self.manifest)
            if (
                not isinstance(data, dict)
                or type(data.get("schema_version")) is not int
                or data.get("schema_version") != SCHEMA_VERSION
            ):
                return [], "invalid_manifest"
            entries = data.get("candidates")
            if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                return [], "invalid_manifest"
            return entries, None
        except (OSError, ValueError, TypeError, RecursionError):
            return [], "invalid_manifest"

    def match_exact(self, request, *, implementation_sha256, binary_sha256):
        entries, failure = self.entries()
        if failure:
            return None, failure
        failure = "missing_evidence"
        expected = json.loads(canonical_json(asdict(request)))
        matches = []
        for entry in entries:
            try:
                if entry.get("request") != expected:
                    failure = "scope_mismatch"
                    continue
                if (
                    entry.get("implementation_sha256") != implementation_sha256
                    or entry.get("binary_sha256") != binary_sha256
                ):
                    failure = "stale_evidence"
                    continue
                if entry.get("numerical_status") != "pass":
                    failure = "failed_parity"
                    continue
                if entry.get("performance_status") != "qualified":
                    failure = "regression_blocked"
                    continue
                if (
                    entry.get("approved_for_runtime") is not True
                    or not isinstance(entry.get("reviewed_by"), str)
                    or not entry["reviewed_by"]
                ):
                    failure = "unapproved_evidence"
                    continue
                candidate = KernelCandidate(**entry["candidate"])
                if (
                    candidate.kernel_id != KERNEL_ID
                    or candidate.semantic_version != "1.0.0"
                    or candidate.implementation_sha256 != implementation_sha256
                    or candidate.binary_sha256 != binary_sha256
                    or type(candidate.threadgroup_size) is not int
                    or candidate.threadgroup_size not in THREADGROUPS
                ):
                    failure = "invalid_manifest"
                    continue
                # Evidence is a sibling content-addressed file. No paths or executable hooks.
                sha = candidate.evidence_sha256
                if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                    failure = "invalid_manifest"
                    continue
                root = self.manifest.resolve().parent
                proof_path = root / (sha + ".json")
                if proof_path.resolve().parent != root:
                    failure = "invalid_manifest"
                    continue
                proof = read_json(proof_path)
                if digest_json(proof) != sha:
                    failure = "stale_evidence"
                    continue
                if (
                    type(proof.get("schema_version")) is not int
                    or proof.get("schema_version") != SCHEMA_VERSION
                    or proof.get("request") != expected
                    or proof.get("implementation_sha256") != implementation_sha256
                    or proof.get("binary_sha256") != binary_sha256
                ):
                    failure = "stale_evidence"
                    continue
                trials = [
                    r
                    for r in proof.get("results", [])
                    if r.get("threadgroup_size") == candidate.threadgroup_size
                ]
                if len(trials) != 1 or not all(
                    trials[0].get(k) is True for k in ("correct", "stable", "no_regression")
                ):
                    failure = "regression_blocked"
                    continue
                profile = trials[0]["profile"]
                reference = proof.get("reference_profile", {})
                if not valid_profile(profile) or not valid_profile(reference):
                    failure = "missing_evidence"
                    continue
                mean = statistics.mean(profile["samples_ms"])
                if (
                    statistics.pstdev(profile["samples_ms"]) / mean > 0.25
                    or profile["median_ms"] > reference["median_ms"] * 1.05
                ):
                    failure = "regression_blocked"
                    continue
                error = trials[0].get("max_abs_error")
                if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
                    failure = "missing_evidence"
                    continue
                matches.append(candidate)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
                failure = "invalid_manifest"
        return (min(matches, key=lambda c: c.candidate_id), None) if matches else (None, failure)


def valid_profile(profile):
    if (
        not isinstance(profile, dict)
        or profile.get("status") != "measured"
        or profile.get("timing_scope") != "synchronized_wall"
        or profile.get("methodology", {}).get("synchronized") is not True
    ):
        return False
    samples = profile.get("samples_ms")
    if (
        not isinstance(samples, list)
        or not 3 <= len(samples) <= 1000
        or profile.get("sample_count") != len(samples)
        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)
    ):
        return False
    median = profile.get("median_ms")
    return (
        type(median) in (int, float)
        and math.isfinite(median)
        and math.isclose(median, statistics.median(samples), rel_tol=1e-12)
    )
