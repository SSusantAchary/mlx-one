"""Backend-specific P1 evidence contracts. No device imports or implicit downloads."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from mlx_one.inference_baseline import TOLERANCES
from mlx_one.inference_qualification import validate_integrity
from mlx_one.p1_profiles import CHECKPOINTS, profile
from mlx_one.runtime_baseline import file_digest

CORRECTNESS_VERSION = "native-p1-correctness-v2"
QUALIFICATION_STATUSES = frozenset({
    "PASS", "FAIL", "DEFERRED", "HARDWARE_PENDING", "NOT_RUN", "UNSUPPORTED", "BLOCKED",
})
BACKENDS = ("reference_dense", "experimental_dense_batch", "experimental_hybrid_batch", "paged")
TEST_FILE = "tests/test_p1_checkpoint_parity.py"


def backend_requirements(backends=None):
    """Omitted selection retains the conservative legacy dense-and-paged contract."""
    selected = tuple(BACKENDS if backends is None else backends)
    if not selected or set(selected) - set(BACKENDS):
        raise ValueError("explicit supported correctness backends are required")
    return tuple(paged for paged in (False, True)
                 if any((backend == "paged") == paged for backend in selected))


def execution_mode(checkpoint, paged):
    return "paged" if paged else ("experimental_hybrid_batch"
        if CHECKPOINTS[checkpoint].architecture == "lfm2" else "experimental_dense_batch")


def test_nodes(checkpoint, paged):
    return [f"{TEST_FILE}::test_native_p1_serial_dense_paged_full_state"
            f"[{paged}-{page}-{size}-{checkpoint}]"
            for page in (16, 32) for size in (1, 2, 4)]


def correctness_command(checkpoint, paged, directory, executable):
    return [executable, "-m", "pytest", "--no-cov", "-q",
            f"--junitxml={Path(directory).resolve() / 'junit.xml'}",
            *test_nodes(checkpoint, paged)]


def checkpoint_proof(checkpoint, local_path=None):
    """Hash the actual offline checkpoint, including tokenizer and conversion metadata."""
    asset = CHECKPOINTS[checkpoint]
    if local_path is None:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(asset.model_id, revision=asset.revision,
                                       local_files_only=True)
    root = Path(local_path).resolve(strict=True)
    item = profile(checkpoint, mode=execution_mode(checkpoint, False), checkpoint_path=str(root))
    files = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*"))
             if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)}
    if ("config.json" not in files or not any(p.endswith(".safetensors") for p in files)
            or not any(p in files for p in ("tokenizer.json", "tokenizer.model", "vocab.json"))):
        raise ValueError("checkpoint proof requires weights, config and tokenizer")
    return {"checkpoint": checkpoint, "model_id": asset.model_id,
            "source_revision": asset.revision, "revision": item.revision,
            "architecture": asset.architecture, "path": str(root), "files": files}


# One immutable snapshot per test worker; parent rechecks files after the worker exits.
cached_checkpoint_proof = lru_cache(maxsize=6)(checkpoint_proof)


def parity_identity(checkpoint, paged, page, size, asset):
    return {"checkpoint": asset, "execution_mode": execution_mode(checkpoint, paged),
            "attention_backend": "reference", "batch_size": size, "page_tokens": page,
            "test_node": test_nodes(checkpoint, paged)[(0 if page == 16 else 3)
                                                     + (1, 2, 4).index(size)],
            "execution_dtype": "float16", "cache_dtype": "float16",
            "protocol": CORRECTNESS_VERSION}


def validate_native_row(directory, checkpoint, paged, *, source, asset):
    """Validate exact tests, command identity, immutable weights and numerical/state receipts."""
    directory = Path(directory)
    names = {f"parity-{checkpoint}-{paged}-{page}-{size}.json"
             for page in (16, 32) for size in (1, 2, 4)}
    validate_integrity(directory, required={
        "supervision.json", "command.json", "junit.xml", "worker.log", *names,
    })
    command = json.loads((directory / "command.json").read_text())
    argv = command.get("argv", [])
    command_source = {key: value for key, value in source.items() if key != "native_test_sha256"}
    if (not argv or argv != correctness_command(checkpoint, paged, directory, argv[0])
            or command.get("source") != command_source):
        raise ValueError("correctness command or source identity mismatch")
    supervision = json.loads((directory / "supervision.json").read_text())
    if (supervision.get("status") != "completed"
            or supervision.get("worker_log_sha256") != file_digest(directory / "worker.log")):
        raise ValueError("native correctness log is missing, changed or incomplete")
    try:
        tests = list(ET.parse(directory / "junit.xml").getroot().iter("testcase"))
    except ET.ParseError as exc:
        raise ValueError("native correctness JUnit receipt is malformed") from exc
    expected = {node.split("::")[-1] for node in test_nodes(checkpoint, paged)}
    if (len(tests) != 6 or {test.get("name") for test in tests} != expected
            or any(list(test) for test in tests)
            or any(test.get("classname") != "tests.test_p1_checkpoint_parity" for test in tests)):
        raise ValueError("native correctness test identities are missing, skipped or failed")
    for page in (16, 32):
        for size in (1, 2, 4):
            parity = json.loads((directory / f"parity-{checkpoint}-{paged}-{page}-{size}.json"
                                 ).read_text())
            if parity.get("test_identity") != parity_identity(checkpoint, paged, page, size, asset):
                raise ValueError("native parity checkpoint/backend/test identity mismatch")
            scope = parity.get("runtime", {}).get("scope", {})
            if (parity.get("status") != "passed"
                    or parity.get("execution_mode") != execution_mode(checkpoint, paged)
                    or scope.get("execution_path") != execution_mode(checkpoint, paged)
                    or scope.get("revision") != asset["revision"]
                    or scope.get("mlx_version") != source.get("packages", {}).get("mlx")
                    or parity.get("batch_size") != size or parity.get("forced_steps") != 32
                    or (parity.get("atol"), parity.get("rtol")) != TOLERANCES["float16"]
                    or parity.get("greedy_agreement_fraction") != 1
                    or parity.get("prefix_isolation_passed") is not True
                    or parity.get("seeded_rng_isolation") is not True
                    or parity.get("cleanup", {}).get("active_handles") != 0
                    or parity.get("cleanup", {}).get("reserved_bytes") != 0):
                raise ValueError("native parity/state artifacts are incomplete")
            for group in ("comparisons", "component_comparisons", "prefix_restoration"):
                rows = parity.get(group, [])
                if not rows or any(row.get("within_tolerance") is not True
                                   or row.get("greedy_agreement", True) is not True
                                   or any(not isinstance(row.get(key), (int, float))
                                          or not math.isfinite(row[key]) or row[key] < 0
                                          for key in ("max_error", "mean_error")) for row in rows):
                    raise ValueError("native numerical comparisons are missing or failed")
