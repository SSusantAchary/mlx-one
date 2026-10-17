"""Opt-in native math parity/profiling cohorts over the existing offline supervisor."""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

# This is repository tooling, also invoked as an isolated script by the supervisor.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_one.inference_baseline import source_provenance
from mlx_one.inference_qualification import validate_integrity
from mlx_one.native_qualification import preflight
from mlx_one.p0_engine_qualification import (
    atomic_json,
    case_inventory,
    digest,
    exact_prompt,
    utc,
    validate_assets,
)
from qualification.p01_math import (
    WIDTHS,
    await_prerequisites,
    canonical_digest,
    experiment_inventory,
    validate_worker_result,
    write_integrity,
)
from qualification.p01_math import run_task as supervise_task

VERSION = "native-math-experiment-v3"
ORDER = ("openai-community/gpt2", "mlx-community/Qwen3-0.6B-8bit",
         "mlx-community/Qwen2.5-0.5B-Instruct-8bit", "mlx-community/LFM2-350M-4bit",
         "mlx-community/Qwen3.5-0.8B-4bit")


def tool_sources():
    return {str(path.resolve()): digest(path) for path in (
        Path(__file__), Path(__file__).with_name("p01_math.py"),
        Path(__file__).with_name("p01_native_cases.py"),
        Path(__file__).with_name("p01_prompts.py"))}


def validate_sources(expected, source):
    current = source_provenance(Path.cwd())
    if (current["source_sha256"] != source["source_sha256"]
            or current["packages"] != source["packages"]
            or current["software"] != source["software"]):
        raise ValueError("runtime source or execution environment changed")
    if any(digest(path) != checksum for path, checksum in expected.items()):
        raise ValueError("qualification tooling changed")


def policy_settings(task):
    data = {"attention": "reference", "projection": "reference", "logits": "all",
            "hybrid": "reference", "prefill_chunk_size": task["chunk"], "enable_compile": False}
    if task["route"] == "optimized":
        data.update(task["operators"], logits="last_only")
    return data


def unavailable_operator(task, metadata):
    if task.get("operators", {}).get("attention") == "mlx_sdpa":
        from mlx_one.engine.kernel_dispatch import math_availability

        status = math_availability(metadata["architecture"], quantized=bool(
            metadata.get("quantization")), configured_fp16=True)["fused_attention"]
        if status["availability"] == "unavailable":
            return status["reason"]
    return None


def validate_result(result, task, metadata, source):
    if result.get("schema_version") != VERSION:
        raise ValueError("native math evidence schema mismatch")
    if result.get("source") != source or result.get("policy") != policy_settings(task):
        raise ValueError("native math source or policy mismatch")
    identity = result.get("identity", {})
    required = {"checkpoint": metadata["model_id"], "revision": metadata["revision"],
                "assets_sha256": canonical_digest(metadata["assets"]),
                "prompt_tokens": task["prompt_tokens"],
                "source_sha256": source["source_sha256"], "execution_dtype": "float16",
                "kv_dtype": "float16", "batch": 1, "prefix_reuse": False,
                "profile": task["profile"]}
    if any(identity.get(key) != value for key, value in required.items()):
        raise ValueError("native math checkpoint, execution or shape mismatch")
    if len(identity.get("prompt_ids", [])) != task["prompt_tokens"]:
        raise ValueError("native math exact rendered tokens missing")
    if task["workload"] == "release":
        required = {c["case_id"] for c in case_inventory() if c["suite"] in task["suites"]
                    and c["mandatory"]}
        rows = result.get("release_cases", [])
        actual = {c["case_id"]: c for c in rows}
        if (not required or len(actual) != len(rows) or required - actual.keys()
                or any(actual[case]["status"] != "PASS" for case in required)):
            raise ValueError("missing/failed native release correctness or lifecycle cases")
        if any(result.get("ownership", {}).get(key) != 0 for key in
               ("active_handles", "reserved_bytes", "allocated_bytes")):
            raise ValueError("release cleanup incomplete")
        return
    if task["workload"] == "parity":
        parity = result.get("native_parity", {})
        if (parity.get("status") != "PASS" or not parity.get("logits")
                or not parity.get("components") or not parity.get("full_prefill_components")
                or not parity.get("forced_tokens")):
            raise ValueError("missing or mismatched native mathematical parity")
        metrics = [*parity["logits"], *parity["components"],
                   *parity["full_prefill_components"], parity.get("independent_full_prefill", {})]
        if any(metric.get("within_tolerance") is not True for metric in metrics):
            raise ValueError("native numerical errors exceed unchanged tolerance")
        if (not all(metric.get("greedy_agreement") is True for metric in parity["logits"])
                or parity["independent_full_prefill"].get("greedy_agreement") is not True
                or parity.get("token_ids") != parity.get("target_token_ids")):
            raise ValueError("native greedy token parity failed")
        shapes = parity.get("observed_outputs", {})
        dispatch = parity.get("dispatch", [])
        requested = policy_settings(task)
        if requested["attention"] == "mlx_sdpa" and task["prompt_tokens"] > 1:
            if not any(e.get("operation") == "attention" and e.get("selected") == "mlx_sdpa"
                       for e in dispatch):
                raise ValueError("requested fused attention did not demonstrably execute")
        if requested["projection"] == "mlx_qmm":
            if not any(e.get("operation") == "projection" and e.get("selected") == "mlx_qmm"
                       for e in dispatch):
                raise ValueError("requested packed QMM did not demonstrably execute")
        for name in ("reference", "target", "full_reference"):
            records = shapes.get(name, [])
            if not records or any(r.get("logits_dtype") != "float16" for r in records):
                raise ValueError("evaluated output shape and precision evidence missing")
            for record in records:
                wanted = (1 if name == "target" and policy_settings(task)["logits"] ==
                          "last_only" else record["input_tokens"])
                if (record["logits"][:2] != [1, wanted]
                        or record["hidden"][:2] != [1, record["input_tokens"]]):
                    raise ValueError("logits selection changed hidden state or projected all rows")
        if any(result.get("ownership", {}).get(key) != 0 for key in
               ("active_handles", "reserved_bytes", "allocated_bytes")):
            raise ValueError("parity cleanup incomplete")
        return
    if task["workload"] == "controlled":
        validate_controlled_result(result, task)
        return
    # Reuse token, precision, ownership and repetition checks with profiling kept
    # explicitly separate; the comparison layer rejects instrumented identities.
    view = {**result, "identity": {**result["identity"], "profile": False}}
    validate_worker_result(view, task, metadata, source)


def validate_controlled_result(result, task, *, require_native_ownership=True):
    fixture = task.get("forced_tokens")
    if not isinstance(fixture, list) or not fixture:
        raise ValueError("declared forced-token fixture required")
    if (result["identity"].get("measurement_kind") != "synchronized_prepared_history_forwards"
            or result["identity"].get("output_fixture") != fixture):
        raise ValueError("controlled decode must remain separate from generated-token timings")
    rows = result.get("measurements", [])
    if len(rows) != task["repetitions"]:
        raise ValueError("incomplete controlled repetitions")
    for row in rows:
        if (row.get("forced_tokens") != fixture or row.get("output_kind") != "forced_not_generated"
                or row.get("prepared_history_position") != task["prompt_tokens"]
                or row.get("committed_position") != task["prompt_tokens"] + len(fixture)
                or row.get("actual_batch_sizes") != [1]
                or row.get("actual_prefill_tokens") != task["prompt_tokens"]
                or len(row.get("synchronized_decode_forward_seconds", [])) != len(fixture)):
            raise ValueError("controlled tokens, positions, actual forwards or timing incomplete")
    if require_native_ownership and any(result.get("ownership", {}).get(key) != 0 for key in
           ("active_handles", "reserved_bytes", "allocated_bytes")):
        raise ValueError("controlled request cleanup incomplete")


def validate_parity_receipt(path, task, metadata, source, hardware):
    path = Path(path).resolve()
    validate_integrity(path.parent, required={path.name})
    result = json.loads(path.read_text())
    parent = path.parent.parent
    validate_integrity(parent, required={"supervision.json", "manifest.json"})
    receipt = json.loads((parent / "supervision.json").read_text())
    manifest = json.loads((parent / "manifest.json").read_text())
    if (receipt.get("status") != "completed" or result.get("status") != "PASS"
            or manifest.get("task", {}).get("workload") != "parity"
            or manifest.get("task", {}).get("profile") is not False
            or manifest["native"]["environment"]["preflight"]["hardware"] != hardware
            or result.get("identity", {}).get("assets_sha256") != canonical_digest(
                metadata["assets"])
            or result.get("identity", {}).get("prompt_tokens") != task["prompt_tokens"]):
        raise ValueError("native parity checkpoint, shape or hardware mismatch")
    validate_result(result, {**task, "workload": "parity"}, metadata, source)
    if (task.get("forced_tokens")
            and result["native_parity"]["forced_tokens"] != task["forced_tokens"]):
        raise ValueError("parity proof does not cover the declared forced continuation")
    return {"path": str(path), "sha256": digest(path)}


def worker(manifest_path, destination):
    from qualification.p01_native_cases import (
        controlled_forward,
        numerical_parity,
        observed_generate,
        prepare_cases,
        write_profile,
    )

    manifest = json.loads(Path(manifest_path).read_text())
    validate_sources(manifest["tool_sources"], manifest["source"])
    if digest(__file__) != manifest["tool_sha256"]:
        raise ValueError("native worker source changed")
    native, task = manifest["native"], manifest["task"]
    metadata = native["metadata"]
    actual = validate_assets(metadata["model_id"], metadata["snapshot_path"], metadata["revision"])
    if actual["assets"] != metadata["assets"]:
        raise ValueError("pinned checkpoint assets changed")
    policy_class = import_module("mlx_one.engine.optimization_policy").NativeMathPolicy
    policy = policy_class(**policy_settings(task))
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "heartbeat.json"
    atomic_json(heartbeat, {"status": "running", "started": time.monotonic(),
                            "timeout": 120, "phase": "load"})
    cases = prepare_cases(native, root, policy=policy, route=task["route"])
    result = {"schema_version": VERSION, "source": manifest["source"],
              "policy": policy_settings(task), "status": "running", "measurements": [],
              "correctness_passed": False, "qualification_promoted": False}
    capture_active = False
    try:
        if task.get("prompt_manifest"):
            from qualification.p01_prompts import resolve_prompt

            prompt, tokens = resolve_prompt(task["prompt_manifest"], task["prompt_manifest_sha256"],
                                            metadata, cases.bundle.tokenizer, task["prompt_tokens"])
        else:
            prompt = exact_prompt(cases.bundle.tokenizer, task["prompt_tokens"])
            tokens = list(cases.bundle.tokenizer.encode(prompt))
        result["identity"] = {"checkpoint": metadata["model_id"], "revision": metadata["revision"],
            "assets_sha256": canonical_digest(metadata["assets"]),
            "tokenizer_identity": cases.bundle.tokenizer_identity, "prompt_ids": tokens,
            "prompt_tokens": len(tokens), "output_fixture": None, "execution_dtype": "float16",
            "kv_dtype": "float16", "source_sha256": manifest["source"]["source_sha256"],
            "environment": {"hardware": native["environment"]["preflight"]["hardware"],
                            "packages": manifest["source"]["packages"],
                            "software": manifest["source"]["software"]},
            "context_limit": cases.bundle.context_length, "seed": 0,
            "prefix_reuse": False, "batch": 1, "profile": task["profile"],
            "measurement_kind": "python_engine_model_token_intervals"}
        atomic_json(root / "prompt.json", {"text": prompt, "token_ids": tokens})
        if task["workload"] == "release":
            for suite in task["suites"]:
                atomic_json(heartbeat, {"status": "running", "started": time.monotonic(),
                                        "timeout": 600, "phase": suite})
                getattr(cases, suite)()
                result["release_cases"] = cases.rows["cases"]
                atomic_json(root / "experiment.json", result)
                if any(row["status"] == "FAIL" for row in result["release_cases"]):
                    raise AssertionError("native release failure stops dependent suites")
            result["status"] = "PASS"
            return
        if task.get("correctness_receipt"):
            proof = task["correctness_receipt"]
            if digest(proof["path"]) != proof["sha256"]:
                raise ValueError("native parity proof changed after admission")
            validate_parity_receipt(proof["path"], task, metadata, manifest["source"],
                                    native["environment"]["preflight"]["hardware"])
            result["correctness_passed"] = True
            result["correctness_receipt"] = proof
        if task["workload"] == "parity":
            atomic_json(heartbeat, {"status": "running", "started": time.monotonic(),
                                    "timeout": 600, "phase": "parity"})
            result["native_parity"] = numerical_parity(cases, tokens, policy=policy,
                chunk=task["chunk"], artifact=root / "logits.npz",
                continuation=task.get("forced_tokens", (1, 2, 3, 4)))
            result["status"] = result["native_parity"]["status"]
            result["correctness_passed"] = result["status"] == "PASS"
            return
        if task["workload"] == "controlled":
            result["identity"].update(output_fixture=task["forced_tokens"],
                                     measurement_kind="synchronized_prepared_history_forwards")
            atomic_json(heartbeat, {"status": "running", "started": time.monotonic(),
                                    "timeout": 600, "phase": "controlled_warmup"})
            controlled_forward(cases, tokens, task["forced_tokens"], policy=policy)
            for repetition in range(task["repetitions"]):
                atomic_json(heartbeat, {"status": "running", "started": time.monotonic(),
                                        "timeout": 600, "phase": "controlled_measurement"})
                cases.mx.synchronize()
                cases.mx.reset_peak_memory()
                started = time.monotonic()
                value = controlled_forward(cases, tokens, task["forced_tokens"], policy=policy)
                result["measurements"].append({**value, "repetition": repetition,
                    "started": started, "finished": time.monotonic(), "instrumented": False,
                    "peak_mlx_bytes": cases.mx.get_peak_memory(),
                    "active_mlx_bytes": cases.mx.get_active_memory(),
                    "cached_mlx_bytes": cases.mx.get_cache_memory(), "rss_peak_bytes": None})
                atomic_json(root / "experiment.json", result)
            result["status"] = "PASS"
            return
        atomic_json(heartbeat, {"status": "running", "started": time.monotonic(),
                                "timeout": 600, "phase": "warmup"})
        warm, warm_observer = observed_generate(cases, prompt, capture_state=True)
        result["identity"]["output_fixture"] = warm["observer"]["token_ids"]
        result["observed_logits_shape"] = warm_observer.logits_shape
        result["components"] = warm_observer.components
        result["native_load_seconds"] = cases.load_seconds
        result["math_initialization_seconds"] = cases.math_initialization_seconds
        for repetition in range(task["repetitions"]):
            atomic_json(heartbeat, {"status": "running", "started": time.monotonic(),
                                    "timeout": 600, "phase": "measurement"})
            cases.mx.synchronize()
            cases.mx.reset_peak_memory()
            if task["capture"]:
                cases.mx.metal.start_capture(str(root / f"capture-{repetition}.gputrace"))
                capture_active = True
            started = time.monotonic()
            with (root / f"phases-{repetition}.jsonl").open("w") as trace:
                value, observer = observed_generate(cases, prompt,
                                                    profile=task["profile"], trace=trace)
            cases.mx.synchronize()
            if capture_active:
                cases.mx.metal.stop_capture()
                capture_active = False
            finished = time.monotonic()
            timing = observer.summary()
            if timing["token_ids"] != result["identity"]["output_fixture"]:
                raise AssertionError("fresh-state greedy tokens changed")
            prefill_ms = timing["forward_ms"]["prefill"]
            result["measurements"].append({"repetition": repetition, "started": started,
                "finished": finished, "generated_tokens": value["generation_tokens"],
                "prefill_tps": len(tokens) / (prefill_ms / 1000),
                "decode_tps": timing["decode_tokens_per_second"],
                "engine_wall_seconds": value["engine_wall_seconds"],
                "peak_mlx_bytes": cases.mx.get_peak_memory(),
                "active_mlx_bytes": cases.mx.get_active_memory(),
                "cached_mlx_bytes": cases.mx.get_cache_memory(), "rss_peak_bytes": None,
                "timing": timing, "gpu_kernel_count": None, "instrumented": task["profile"]})
            if task["profile"]:
                write_profile(root / f"profile-{repetition}.json", observer)
            atomic_json(root / "experiment.json", result)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", reason=str(error), error_type=type(error).__name__)
        raise
    finally:
        if capture_active:
            cases.mx.metal.stop_capture()
        cases.close()
        result["ownership"] = cases.engine._cache_manager.stats().to_dict()
        atomic_json(root / "experiment.json", result)
        atomic_json(heartbeat, {"status": "finished"})
        write_integrity(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--model", action="append")
    parser.add_argument("--route", nargs="+", choices=("reference", "optimized"))
    parser.add_argument("--workload",
                        choices=("engine", "parity", "profile", "controlled", "release"),
                        default="engine")
    parser.add_argument("--suite", action="append",
                        choices=("correctness", "streaming", "state", "lifecycle"))
    parser.add_argument("--forced-fixture", type=Path,
                        help="JSON with token_ids for explicitly forced controlled decode")
    parser.add_argument("--prompt-tokens", type=int, nargs="+")
    parser.add_argument("--chunk-tokens", type=int, choices=(256, 512, 1024, 2048))
    parser.add_argument("--attention", choices=("reference", "mlx_sdpa"), default="reference")
    parser.add_argument("--projection", choices=("reference", "mlx_qmm"), default="reference")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--independent-runs", type=int, default=2)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--correctness-proof", type=Path, action="append", default=[])
    parser.add_argument("--after", type=Path, action="append", default=[])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("qualification/results/p01_math"))
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    if not args.baseline or not args.route:
        parser.error("--baseline and explicit --route are required; auto is unavailable")
    if args.capture and args.workload != "profile":
        parser.error("GPU capture is restricted to instrumented profile workers")
    if args.suite and args.workload != "release":
        parser.error("--suite is restricted to native release workers")
    args.prompt_tokens = args.prompt_tokens or ([128] if args.workload == "release" else WIDTHS)
    if args.workload == "release" and args.prompt_tokens != [128]:
        parser.error("release uses its fixed corpus; use parity/engine for length-specific gates")
    allowed_widths = (1, 2, *WIDTHS) if args.workload == "parity" else WIDTHS
    if set(args.prompt_tokens) - set(allowed_widths):
        parser.error("unsupported prompt widths for selected workload")
    fixture = None
    if args.workload in {"controlled", "parity"}:
        if args.forced_fixture is None and args.workload == "controlled":
            parser.error("controlled decode requires an explicit --forced-fixture")
        if args.forced_fixture is not None:
            fixture = json.loads(args.forced_fixture.read_text()).get("token_ids")
            if (not isinstance(fixture, list) or len(fixture) not in {128, 256}
                    or any(type(token) is not int or token < 0 for token in fixture)):
                parser.error("controlled fixture requires exactly 128 or 256 nonnegative token IDs")
    elif args.forced_fixture is not None:
        parser.error("forced fixtures are restricted to controlled raw-model measurements")
    if args.route == ["reference"] and (args.attention != "reference" or
                                       args.projection != "reference"):
        parser.error("reference oracle cannot select optimized operators")
    environment = json.loads((args.baseline / "environment.json").read_text())
    downloads = json.loads((args.baseline / "download_manifest.json").read_text())
    models = [value["metadata"] for value in downloads.values() if value["status"] == "PASS"
              and (args.model is None or value["metadata"]["model_id"] in args.model)]
    models.sort(key=lambda model: ORDER.index(model["model_id"]))
    if not models or args.model and set(args.model) - {m["model_id"] for m in models}:
        parser.error("requested immutable offline assets are incomplete")
    if args.prompt_manifest:
        from qualification.p01_prompts import load_prompts

        if len(models) != 1 or args.workload == "release":
            parser.error("rendered fixtures require one checkpoint and a length-specific workload")
        fixtures = load_prompts(args.prompt_manifest, models[0])
        if set(args.prompt_tokens) - {len(r["token_ids"]) for r in fixtures["records"]}:
            parser.error("requested widths missing from exact rendered-token manifest")
    tasks = experiment_inventory(models, args.prompt_tokens, routes=tuple(args.route),
        independent_runs=args.independent_runs, repetitions=args.repetitions,
        chunk=args.chunk_tokens, profile=args.workload == "profile", allowed_widths=allowed_widths)
    for task in tasks:
        task.update(workload=args.workload, capture=args.capture,
                    operators={"attention": args.attention, "projection": args.projection},
                    diagnostic=args.diagnostic)
        if args.prompt_manifest:
            task.update(prompt_manifest=str(args.prompt_manifest.resolve()),
                        prompt_manifest_sha256=digest(args.prompt_manifest))
        if args.workload == "release":
            task.update(suites=args.suite or ["correctness", "streaming", "state", "lifecycle"])
            task["supervision_timeout_seconds"] = 120 + 600 * len(task["suites"])
        if fixture is not None:
            task.update(forced_tokens=fixture, forced_fixture_sha256=digest(args.forced_fixture))
            if task["prompt_tokens"] + len(fixture) > next(m["configured_context"] for m in
                    models if m["model_id"] == task["checkpoint"]):
                task.update(status="NOT_SUPPORTED",
                            reason="prompt plus forced output exceeds context")
        model = next(m for m in models if m["model_id"] == task["checkpoint"])
        unavailable = unavailable_operator(task, model)
        if unavailable:
            task.update(status="NOT_SUPPORTED", reason=unavailable)
        task["id"] = canonical_digest({k: v for k, v in task.items() if k != "id"})[:24]
    root = args.output.resolve() / str(time.time_ns())
    root.mkdir(parents=True)
    source, tools = source_provenance(Path.cwd()), tool_sources()
    manifest = {"schema_version": VERSION, "source": source, "tool_sources": tools,
        "tasks": tasks, "status": "QUEUED" if args.run else "NOT_RUN", "timestamp": utc(),
        "qualification_promoted": False}
    atomic_json(root / "manifest.json", manifest)
    write_integrity(root)
    print(root, flush=True)
    if not args.run:
        return 0
    if all(task["status"] == "NOT_SUPPORTED" for task in tasks):
        manifest.update(status="completed", reason="no supported model execution selected")
        atomic_json(root / "manifest.json", manifest)
        write_integrity(root)
        return 0
    try:
        await_prerequisites(args.after)
        validate_sources(tools, source)
        import_module("mlx_one.engine.optimization_policy")  # Capability check before GPU work.
        observed = preflight(root / "preflight")
        manifest.update(status="running", preflight=observed)
        for index, task in enumerate(tasks):
            if task["status"] != "NOT_RUN":
                continue
            validate_sources(tools, source)
            model = next(m for m in models if m["model_id"] == task["checkpoint"])
            if task["workload"] in {"engine", "controlled"} and not args.diagnostic:
                proofs = []
                for path in args.correctness_proof:
                    try:
                        proofs.append(validate_parity_receipt(path, task, model, source,
                                                              observed["hardware"]))
                    except ValueError:
                        continue
                if not proofs:
                    task.update(status="BLOCKED", reason="matching native parity proof missing")
                    continue
                task["correctness_receipt"] = proofs[0]
            tasks[index] = supervise_task(root, task, model, environment, source, observed,
                worker_entry=__file__, validator=validate_result, tool_sources=tools,
                schema_version=VERSION)
            atomic_json(root / "manifest.json", manifest)
            write_integrity(root)
            print(json.dumps(tasks[index]), flush=True)
            if tasks[index]["status"] != "PASS":
                raise RuntimeError("native failure stops expansion of this cohort")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="BLOCKED", reason=str(error), error_type=type(error).__name__)
        raise
    finally:
        atomic_json(root / "manifest.json", manifest)
        write_integrity(root)
    return int(any(t["status"] not in {"PASS", "NOT_SUPPORTED"} for t in tasks))


if __name__ == "__main__":
    raise SystemExit(main())
