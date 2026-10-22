"""Five-checkpoint reference Engine qualification; existing native evidence machinery."""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

from mlx_one.inference_baseline import source_provenance, supervise
from mlx_one.inference_qualification import write_integrity
from mlx_one.native_qualification import preflight

SCHEMA = "native-p0-engine-v2"
MODELS = {
    "openai-community/gpt2": ("gpt2", None),
    "mlx-community/Qwen3-0.6B-8bit": ("qwen3", 8),
    "mlx-community/Qwen2.5-0.5B-Instruct-8bit": ("qwen2", 8),
    "mlx-community/LFM2-350M-4bit": ("lfm2", 4),
    "mlx-community/Qwen3.5-0.8B-4bit": ("qwen3_5", 4),
}
CACHED_PINS = {
    "openai-community/gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "mlx-community/Qwen3-0.6B-8bit": "11de96878523501bcaa86104e3c186de07ff9068",
    "mlx-community/LFM2-350M-4bit": "18dc72abf3b2337f9123cfd6eeeb58dfa7947066",
}
SUITES = ("correctness", "streaming", "state", "lifecycle", "admission", "benchmark")
GROUPS = {"A": ("startup", 14), "B": ("correctness", 6), "C": ("streaming", 10),
          "D": ("state", 10), "E": ("state", 10), "F": ("lifecycle", 12),
          "G": ("lifecycle", 10)}
STATUSES = {"PASS", "FAIL", "SKIP", "NOT_RUN", "BLOCKED", "NOT_SUPPORTED"}
PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken",
            "*.jinja", "*.jinja2", "LICENSE*", "README.md")


def utc():
    return datetime.now(timezone.utc).isoformat()


def exact_prompt(tokenizer, target_tokens):
    """Build and verify a stable rendered token count, never a character approximation."""
    if target_tokens < 1:
        raise ValueError("positive prompt token count required")
    for unit in (" x", " a", " hello", " local", " 1", "\n", "a"):
        prompt = unit * target_tokens
        ids = tokenizer.encode(prompt)
        if len(ids) == target_tokens:
            return prompt
        if len(ids) > target_tokens:
            prompt = tokenizer.decode(ids[:target_tokens], skip_special_tokens=False)
            if len(tokenizer.encode(prompt)) == target_tokens:
                return prompt
    raise ValueError("cannot construct an exact tokenizer-roundtrip prompt")


def qualification_exit(rows, selected, suites, *, prompt_tokens=None):
    """A conditional release recommendation must never hide a requested failed gate."""
    requested = {"startup", *suites}
    benchmark_cases = {f"H{n}" for n in prompt_tokens} if prompt_tokens is not None else None
    return int(any(row["model_id"] in selected and row["suite"] in requested
                   and (row["suite"] != "benchmark" or benchmark_cases is None
                        or row["case_id"] in benchmark_cases)
                   and row["mandatory"] and row.get("applicable", True)
                   and row["status"] != "PASS" for row in rows))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def headers(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 100 * 1024**2:
            raise ValueError("oversized safetensors header")
        tensors = {k: v for k, v in json.loads(stream.read(size)).items() if k != "__metadata__"}
        payload = Path(path).stat().st_size - 8 - size
        widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "U32": 4, "I32": 4,
                  "I64": 8, "U64": 8, "I16": 2, "U16": 2, "U8": 1, "I8": 1, "BOOL": 1}
        for name, meta in tensors.items():
            start, end = meta["data_offsets"]
            if (not 0 <= start <= end <= payload or meta["dtype"] not in widths or
                    end - start != math.prod(meta["shape"]) * widths[meta["dtype"]]):
                raise ValueError(f"invalid or truncated tensor payload: {name}")
        return tensors


def validate_assets(model_id, root, revision):
    from mlx_one.text.loading import _quantization_config

    root = Path(root)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("immutable checkpoint revision required")
    config = json.loads((root / "config.json").read_text())
    architecture, bits = MODELS[model_id]
    if config.get("model_type") != architecture:
        raise ValueError("checkpoint architecture mismatch")
    quantization = _quantization_config(config)
    if quantization.get("bits") != bits:
        raise ValueError("checkpoint weight precision mismatch")
    if not (root / "tokenizer_config.json").is_file():
        raise ValueError("missing tokenizer_config.json")
    if not ((root / "tokenizer.json").is_file() or
            ((root / "vocab.json").is_file() and (root / "merges.txt").is_file()) or
            (root / "tokenizer.model").is_file()):
        raise ValueError("missing validated tokenizer assets")
    tensor_meta = {}
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError("missing safetensors weights")
    for path in files:
        part = headers(path)
        if set(part) & set(tensor_meta):
            raise ValueError("duplicate tensors across shards")
        tensor_meta.update(part)
    for index in root.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text())["weight_map"]
        for name, filename in mapping.items():
            shard = root / filename
            if shard.parent != root or not shard.is_file() or name not in headers(shard):
                raise ValueError(f"missing or invalid indexed tensor: {name}")
        if set(mapping) != set(tensor_meta):
            raise ValueError("weight index does not match complete shard inventory")
    packed, logical, elements = 0, 0, 0
    for name, meta in tensor_meta.items():
        count = math.prod(meta["shape"])
        elements += count
        if name.endswith((".scales", ".biases")):
            continue
        if name.endswith(".weight") and meta["dtype"] == "U32":
            if not bits or len(meta["shape"]) != 2:
                raise ValueError("unsupported packed tensor")
            packed += 1
            width = meta["shape"][1] * 32 // bits
            group = quantization["group_size"]
            scales = tensor_meta.get(name[:-6] + "scales")
            biases = tensor_meta.get(name[:-6] + "biases")
            expected = [meta["shape"][0], width // group]
            if (width % group or not scales or not biases or
                    scales["shape"] != expected or biases["shape"] != expected or
                    scales["dtype"] not in {"F16", "BF16", "F32"} or
                    scales["dtype"] != biases["dtype"]):
                raise ValueError(f"invalid affine tensor layout: {name}")
            logical += count * 32 // bits
        elif not name.endswith((".attn.bias", ".attn.masked_bias")):
            logical += count
    if bits and not packed:
        raise ValueError("quantization declared without packed affine modules")
    assets = {str(p.relative_to(root)): {"sha256": digest(p), "bytes": p.stat().st_size}
              for p in sorted(root.rglob("*")) if p.is_file()}
    text = config.get("text_config", config)
    logical_deduplicated = logical
    # Safetensors normally omits tied aliases. Verify and deduplicate any explicit head alias.
    if text.get("tie_word_embeddings") and "lm_head.weight" in tensor_meta:
        for name in ("model.embed_tokens.weight", "transformer.wte.weight",
                     "model.language_model.embed_tokens.weight"):
            if name in tensor_meta and tensor_meta[name]["shape"] == tensor_meta[
                    "lm_head.weight"]["shape"]:
                raise ValueError("explicit tied head requires loader-validated alias identity")
    heads = text.get("num_attention_heads", text.get("n_head"))
    hidden = text.get("hidden_size", text.get("n_embd"))
    intermediates = {meta["shape"][-1] if name.endswith("mlp.c_fc.weight") else meta["shape"][0]
        for name, meta in tensor_meta.items() if name.endswith(("mlp.c_fc.weight",
            "mlp.gate_proj.weight", "feed_forward.w1.weight"))}
    license_info = config.get("license")
    card = root / "README.md"
    if card.is_file() and card.read_text().startswith("---"):
        import yaml

        frontmatter = card.read_text().split("---", 2)
        if len(frontmatter) == 3:
            license_info = (yaml.safe_load(frontmatter[1]) or {}).get("license", license_info)
    metadata = {"model_id": model_id, "revision": revision, "snapshot_path": str(root),
        "architecture": architecture, "task": "text-generation", "quantization": quantization,
        "logical_parameters_before_tie_deduplication": logical,
        "logical_parameters": logical_deduplicated,
        "topology": {"layers": text.get("num_hidden_layers", text.get("n_layer")),
            "attention_heads": heads, "kv_heads": text.get("num_key_value_heads", heads),
            "head_dimension": text.get("head_dim", hidden // heads if hidden and heads else None),
            "hidden_dimension": hidden, "intermediate_dimension": text.get("intermediate_size") or
                (next(iter(intermediates)) if len(intermediates) == 1 else None),
            "intermediate_dimensions": sorted(intermediates),
            "vocabulary_size": text.get("vocab_size"), "layer_types": text.get("layer_types"),
            "position_encoding": "learned" if architecture == "gpt2" else "rope"},
        "stored_tensor_elements": elements, "packed_modules": packed,
        "weight_file_bytes": sum(p.stat().st_size for p in files),
        "snapshot_bytes": sum(item["bytes"] for item in assets.values()),
        "floating_execution_policy": "float16", "attention_kv_policy": "float16",
        "tokenizer_identity": {k: v["sha256"] for k, v in assets.items()
                               if "token" in k or k in {"vocab.json", "merges.txt"}},
        "configuration": text, "assets": assets, "tensor_inventory": tensor_meta,
        "license": license_info,
        "configured_context": text.get("max_position_embeddings", text.get("n_positions"))}
    return metadata


def acquire(model_id, *, offline=False, revision=None, root=None, retries=3):
    """Resolve once, download only needed formats, preserve all attempts."""
    from huggingface_hub import HfApi, snapshot_download

    attempts = []
    pin = revision
    for index in range(retries):
        try:
            license_info = None
            if pin is None and not offline:
                info = HfApi().model_info(model_id, files_metadata=True)
                pin = info.sha
                license_info = getattr(info.card_data, "license", None)
                estimated = sum(f.size or 0 for f in info.siblings
                                if any(fnmatch.fnmatch(f.rfilename, p) for p in PATTERNS))
                if shutil.disk_usage(Path.home()).free < estimated * 1.2 + 1024**3:
                    raise OSError("insufficient free disk for pinned snapshot")
            if pin is None:
                pin = CACHED_PINS.get(model_id)
            if pin is None and offline:
                from huggingface_hub import scan_cache_dir

                revisions = {s.commit_hash for repo in scan_cache_dir().repos
                             if repo.repo_id == model_id for s in repo.revisions}
                if len(revisions) == 1:
                    pin = revisions.pop()
            if pin is None:
                raise ValueError("offline mode requires an existing immutable checkpoint pin")
            if root is not None:
                atomic_json(Path(root) / "pins" / (model_id.replace("/", "--") + ".json"),
                            {"model_id": model_id, "revision": pin, "resolved_at": utc()})
            path = snapshot_download(model_id, revision=pin, local_files_only=True,
                                     allow_patterns=list(PATTERNS))
            try:
                metadata = validate_assets(model_id, path, pin)
            except (OSError, ValueError, KeyError):
                if offline:
                    raise
                path = snapshot_download(model_id, revision=pin, allow_patterns=list(PATTERNS))
                metadata = validate_assets(model_id, path, pin)
            metadata["license"] = license_info or metadata.get("license")
            return {"status": "PASS", "metadata": metadata, "attempts": attempts}
        except Exception as exc:
            attempts.append({"attempt": index + 1, "revision": pin,
                             "error_type": type(exc).__name__, "error": str(exc)})
            # A missing snapshot must reach the online downloader on the next attempt.
            if pin is not None and not offline:
                try:
                    path = snapshot_download(model_id, revision=pin,
                                             allow_patterns=list(PATTERNS))
                    metadata = validate_assets(model_id, path, pin)
                    return {"status": "PASS", "metadata": metadata, "attempts": attempts}
                except Exception as retry_error:
                    attempts.append({"attempt": index + 1, "revision": pin,
                                     "error_type": type(retry_error).__name__,
                                     "error": str(retry_error)})
            if index + 1 < retries and not offline:
                time.sleep(min(2**index, 4))
    return {"status": "BLOCKED", "reason": "BLOCKED_DOWNLOAD", "revision": pin,
            "model_id": model_id, "attempts": attempts}


def case_inventory():
    records = [{"case_id": f"{letter}{index:02}", "suite": suite, "mandatory": True}
               for letter, (suite, count) in GROUPS.items() for index in range(1, count + 1)]
    records.extend({"case_id": f"H{length}", "suite": "benchmark", "mandatory": True}
                   for length in (128, 512, 2048, 8192))
    records.extend({"case_id": f"I{index:02}", "suite": "admission", "mandatory": True}
                   for index in range(1, 7))
    records.append({"case_id": "P01", "suite": "state", "mandatory": False})
    return records


def decide_model(rows):
    required = [r for r in rows if r["mandatory"] and r.get("applicable", True)]
    if any(r["status"] == "FAIL" for r in required):
        return "FAILED"
    if any(r["status"] == "BLOCKED" for r in required):
        return "BLOCKED"
    if not required or any(r["status"] != "PASS" for r in required):
        return "NOT_TESTED"
    return "QUALIFIED"


def release_decision(models, *, shared_blocker=False):
    count = sum(status in {"QUALIFIED", "QUALIFIED_WITH_LIMITATIONS"}
                for status in models.values())
    return "NO_GO" if shared_blocker else (
        "GO" if count == 5 else "CONDITIONAL_GO" if count >= 4 else "NO_GO")


def csv_write(path, rows):
    if not rows:
        rows = [{"status": "NOT_RUN"}]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def aggregate(root, environment, downloads, results):
    rows, metadata, benchmarks, memory, parity, failures = [], [], [], [], [], []
    admission = []
    workers = []
    decisions = {}
    for model_id in MODELS:
        outcome = downloads.get(model_id)
        if outcome and outcome["status"] == "PASS":
            actual = dict(outcome["metadata"])
            for result in results.get(model_id, []):
                for case in result.get("cases", []):
                    if case["status"] != "PASS":
                        continue
                    if case["case_id"] == "A03":
                        actual["native_load"] = case["details"]
                        actual["logical_parameters"] = case["details"][
                            "observed_parameters"]["logical_parameters"]
                    elif case["case_id"] == "A12":
                        actual["native_runtime"] = case["details"]
                    elif case["case_id"] == "E01":
                        actual["evaluated_state_components"] = case["details"]
            metadata.append(actual)
        ledger = {c["case_id"]: {**c, "model_id": model_id, "status": "NOT_RUN",
                                  "reason": "not executed"} for c in case_inventory()}
        for result in results.get(model_id, []):
            if result.get("artifact_index"):
                workers.append({"model_id": model_id, "path": result["artifact_index"],
                                "sha256": result["artifact_index_sha256"]})
            for row in result.get("cases", []):
                if row["status"] not in STATUSES or row["case_id"] not in ledger:
                    raise ValueError("invalid worker result mapping")
                ledger[row["case_id"]].update(row)
            benchmarks.extend(result.get("benchmarks", []))
            admission.extend(result.get("admission", []))
            memory.extend(result.get("memory", []))
            parity.extend(result.get("parity", []))
        if outcome and outcome["status"] != "PASS":
            for row in ledger.values():
                row.update(status=outcome["status"], reason=outcome.get("reason"))
        for row in ledger.values():
            row.setdefault("source", environment["source"])
            row.setdefault("revision", (outcome or {}).get("metadata", {}).get("revision"))
            row.setdefault("timestamp", utc())
            row.setdefault("environment", str(Path(root) / "environment.json"))
        rows.extend(ledger.values())
        failures.extend(r for r in ledger.values() if r["status"] in {"FAIL", "BLOCKED"})
        decisions[model_id] = decide_model(list(ledger.values()))
        if environment.get("policy", {}).get("diagnostic") and decisions[model_id] == "QUALIFIED":
            decisions[model_id] = "NOT_TESTED"
    shared_blocker = any(r.get("shared_blocker", False) for r in failures)
    summary = {"schema_version": SCHEMA, "source": environment["source"],
               "models": decisions, "decision": release_decision(
                   decisions, shared_blocker=shared_blocker), "shared_blocker": shared_blocker,
               "qualification_promoted": False, "timestamp": utc()}
    for filename, value in {"environment.json": environment, "download_manifest.json": downloads,
        "model_metadata.json": metadata, "qualification_results.json": rows,
        "inference_benchmarks.json": benchmarks, "numerical_parity.json": parity,
        "admission_diagnostics.json": admission,
        "failures.json": failures, "summary.json": summary,
        "worker_evidence.json": workers}.items():
        atomic_json(Path(root) / filename, value)
    for filename, value in {"model_metadata.csv": metadata, "qualification_results.csv": rows,
                            "inference_benchmarks.csv": benchmarks,
                            "memory_results.csv": memory}.items():
        csv_write(Path(root) / filename, value)
    write_integrity(root)
    return summary


def run_worker(manifest, suite, directory, ceiling):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "mlx_one.p0_engine_worker", str(manifest), suite,
               str(directory)]
    atomic_json(directory / "command.json", {"argv": command,
        "manifest_sha256": digest(manifest), "timestamp": utc()})
    clock_origin = []
    def observed_clock():
        value = time.monotonic()
        if not clock_origin:
            clock_origin.append(value)
        return value
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        timeout = {"startup": 240, "correctness": 14400, "streaming": 3600,
                   "state": 1800, "lifecycle": 7200, "benchmark": 7200,
                   "admission": 10800, "admission_independent": 900}[suite]
        try:
            supervision = supervise(process, timeout=timeout, memory_limit=ceiling,
                                    clock=observed_clock)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            atomic_json(directory / "supervision.json", {"status": "interrupted",
                                                         "exit_code": process.returncode})
            raise
    supervision["monotonic_origin"] = clock_origin[0] if clock_origin else None
    atomic_json(directory / "supervision.json", supervision)
    path = directory / "results.json"
    result = json.loads(path.read_text()) if path.exists() else {"cases": []}
    if supervision["status"] != "completed":
        completed = {r["case_id"] for r in result["cases"]}
        for case in case_inventory():
            if (case["suite"] == suite or suite == "admission_independent"
                    and case["case_id"] == "I06") and case["case_id"] not in completed:
                result["cases"].append({"case_id": case["case_id"], "status": "BLOCKED",
                    "reason": supervision["status"], "artifact": str(directory / "worker.log")})
    origin = supervision.get("monotonic_origin")
    for measured in result.get("benchmarks", []):
        started, ended = measured.get("started_monotonic"), measured.get("finished_monotonic")
        samples = [sample["rss_bytes"] for sample in supervision.get("rss_samples", [])
                   if origin is not None and started is not None and ended is not None
                   and started <= origin + sample["elapsed_seconds"] <= ended
                   and sample.get("rss_bytes") is not None]
        measured["rss_peak_bytes"] = max(samples) if samples else None
        measured["rss_sampling_interval_seconds"] = .1
        measured["rss_peak_scope"] = "supervised-request-samples" if samples else "unavailable"
    for row in result["cases"]:
        if row["case_id"].startswith("H") and isinstance(row.get("details"), list):
            row["details"] = [measured for measured in result.get("benchmarks", [])
                              if measured.get("case_id") == row["case_id"]]
    if supervision["status"] != "completed":
        for row in result["cases"]:
            if row["status"] == "PASS":
                row.update(observed_status="PASS", status="BLOCKED",
                           reason="worker supervision: " + supervision["status"],
                           shared_blocker=supervision["status"].startswith("memory_watchdog"))
    for row in result["cases"]:
        row["command"] = command
        row["artifact"] = str(path)
    result["supervision"] = supervision
    atomic_json(path, result)
    index = {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()
             and p.name not in {"artifact-index.json", "result-integrity.json"}}
    atomic_json(directory / "artifact-index.json", index)
    write_integrity(directory)
    result["artifact_index"] = str(directory / "artifact-index.json")
    result["artifact_index_sha256"] = digest(directory / "artifact-index.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--preflight", action="store_true")
    selection.add_argument("--download", action="store_true")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--suite", choices=SUITES, action="append")
    parser.add_argument("--prompt-tokens", type=int, choices=(128, 512, 2048, 8192), nargs="+")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Allow selected debug suites; never qualify an incomplete run")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("qualification/results/p0_engine"))
    args = parser.parse_args(argv)
    root = args.output.resolve() / "runs" / str(time.time_ns())
    root.mkdir(parents=True)
    source = source_provenance(Path.cwd())
    environment = {"schema_version": SCHEMA, "source": source, "timestamp": utc(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "python": sys.version, "macos": platform.mac_ver()[0],
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()
                         if d.metadata["Name"]}, "disk_free_bytes": shutil.disk_usage(root).free,
        "policy": {"mode": "reference_dense", "batch": 1, "execution_dtype": "float16",
                   "kv_dtype": "float16", "prefix_reuse": False, "speculation": False,
                   "lifecycle_load_cycles": 20, "generation_cycles": 100,
                   "cache_memory_budget_bytes": 16 * 1024**3,
                   "benchmark_prompt_tokens": args.prompt_tokens or [128, 512, 2048, 8192],
                   "diagnostic": args.diagnostic, "prefill_chunk_tokens": None,
                   "admission_policy": "observed-reclaim-once-v1",
                   "rss_growth_limit_bytes": 128 * 1024**2,
                   "mlx_active_growth_limit_bytes": 32 * 1024**2}}
    downloads, results = {}, {}
    atomic_json(root / "environment.json", environment)
    try:
        observed = preflight(root / "preflight")
        environment["preflight"] = observed
    except Exception as exc:
        environment["preflight"] = {"status": "BLOCKED", "reason": str(exc)}
        aggregate(root, environment, downloads, results)
        print(root, flush=True)
        return 1
    if not args.preflight:
        selected = args.model or list(MODELS)
        for model_id in selected:
            downloads[model_id] = acquire(model_id, offline=args.offline, root=root)
            atomic_json(root / "download_manifest.json", downloads)
        if not args.download:
            suites = args.suite or list(SUITES)
            # A standalone benchmark command must execute its mandatory prerequisites.
            if "benchmark" in suites and not args.diagnostic:
                suites = list(SUITES)
            schedule = [(model, suite) for model in selected
                        for suite in ("startup", *(s for s in suites if s != "benchmark"))]
            if "admission" in suites:
                schedule = [item for pair in schedule for item in
                    ([pair, (pair[0], "admission_independent")]
                     if pair[1] == "admission" else [pair])]
            if "benchmark" in suites:
                schedule.extend((model, "benchmark") for model in selected)
            for model_id, suite in schedule:
                if downloads[model_id]["status"] != "PASS":
                    continue
                slug = model_id.replace("/", "--")
                manifest = root / "models" / slug / "manifest.json"
                if not manifest.exists():
                    atomic_json(manifest, {"environment": environment,
                        "metadata": downloads[model_id]["metadata"]})
                results.setdefault(model_id, [])
                startup_failed = any(r["case_id"].startswith("A") and r["status"] != "PASS"
                    for result in results[model_id] for r in result["cases"])
                if startup_failed and suite != "benchmark":
                    continue
                if source_provenance(Path.cwd())["source_sha256"] != source["source_sha256"]:
                    raise RuntimeError("source changed during qualification cohort")
                if suite == "benchmark" and not args.diagnostic:
                    mandatory = [c for c in case_inventory() if c["mandatory"]
                                 and c["suite"] != "benchmark"]
                    passed = {r["case_id"] for result in results[model_id]
                              for r in result["cases"] if r["status"] == "PASS" or
                              (r["status"] == "NOT_SUPPORTED" and
                               not r.get("applicable", True))}
                    if not all(c["case_id"] in passed for c in mandatory):
                        results[model_id].append({"cases": [{"case_id": c["case_id"],
                            "status": "BLOCKED", "reason": "mandatory prerequisites missing"}
                            for c in case_inventory() if c["suite"] == "benchmark"]})
                        aggregate(root, environment, downloads, results)
                        continue
                outcome = run_worker(manifest, suite, manifest.parent / suite,
                                     observed["process_ceiling_bytes"])
                results[model_id].append(outcome)
                aggregate(root, environment, downloads, results)
                print(json.dumps({"model": model_id, "suite": suite,
                    "statuses": {s: sum(r["status"] == s for r in outcome["cases"])
                                 for s in sorted(STATUSES)}}), flush=True)
                if outcome["supervision"]["status"].startswith("memory_watchdog"):
                    print(root, flush=True)
                    return 1
    summary = aggregate(root, environment, downloads, results)
    # Stable summary pointers; immutable cohort remains the source of every artifact.
    atomic_json(args.output / "latest.json", {"run": str(root), "summary": summary})
    for path in root.iterdir():
        if path.suffix == ".json" and path.name not in {"checksums.json", "result-integrity.json"}:
            atomic_json(args.output / path.name, json.loads(path.read_text()))
        elif path.suffix == ".csv":
            destination = args.output / path.name
            temporary = destination.with_suffix(".csv.tmp")
            shutil.copyfile(path, temporary)
            os.replace(temporary, destination)
    write_integrity(args.output)
    print(root, flush=True)
    if args.preflight or args.download:
        return int(any(v["status"] != "PASS" for v in downloads.values()))
    rows = json.loads((root / "qualification_results.json").read_text())
    return qualification_exit(rows, args.model or list(MODELS), args.suite or list(SUITES),
                              prompt_tokens=args.prompt_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
