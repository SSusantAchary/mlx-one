"""One isolated model/configuration process for the native baseline harness."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import socket
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path

from mlx_one.inference_baseline import BaselineError, BaselineProfile, summarize, validate_identity


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def qualification_evidence(output, *, feature, scope, parity, gate=None, **metadata):
    """Produce checksummed native evidence for later explicit registry promotion."""
    from mlx_one.schemas import ArtifactRef

    artifacts = []
    for name in ("manifest.json", "checkpoint.json", "parity.json", "summary.json",
                 "runtime.json", "cleanup.json", "forward-traces.jsonl", "gate-c.json",
                 "requests.jsonl", "requests.csv", "workload.json", "gate-d.json", "gate-e.json",
                 "gate-f.json", "quality.json", "retrieval.json", "time-series.jsonl",
                 "memory-reconciliation.json", "soak-probes.json", "gate-hybrid-performance.json",
                 "quality-reference.json", "pressure-recovery.json",
                 "quantized-attention-correctness.json", "held-out-quality.json", "traffic.json",
                 "tokenization.json", "actor-events.jsonl", "fairness.json"):
        path = output / name
        if path.is_file():
            reference = ArtifactRef(uri=str(path.resolve()), kind="native-inference-result",
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    media_type="application/json")
            artifacts.append(reference.to_dict())
    result = {"schema_version": "native-inference-qualification-v1", "backend": "mlx-one-native",
              "scope": scope, "feature": feature, "status": parity["status"],
              "correctness_passed": parity["status"] == "passed",
              "performance_qualified": gate is not None and gate.get("status") == "passed",
              "artifacts": artifacts, "qualification_promoted": False, **metadata}
    dependency = output / "quality-reference.json"
    if dependency.is_file():
        reference = json.loads(dependency.read_text())
        artifacts.append(ArtifactRef(uri=reference["path"], kind="native-quality-evidence",
                                     sha256=reference["sha256"]).to_dict())
        artifacts.extend(reference["proof"].get("artifacts", []))
    write_json(output / f"evidence-{feature}.json", result)


def build_messages(bundle, target, *, suffix=""):
    from mlx_one.inference_baseline import deterministic_prompt

    # Shared workload helper from the existing cache HTTP harness.
    overhead = len(bundle.tokenizer.encode(bundle.chat_template.render([
        {"role": "user", "content": suffix}
    ], enable_thinking=False)))
    if target <= overhead + 32:
        raise BaselineError("requested prompt size is smaller than the chat template")
    content = deterministic_prompt(bundle.tokenizer, target - overhead)
    messages = [{"role": "user", "content": content + suffix}]
    prompt = bundle.chat_template.render(messages, enable_thinking=False)
    # Token merges at the template boundary can change length by a few tokens.
    while len(bundle.tokenizer.encode(prompt)) > target:
        ids = bundle.tokenizer.encode(content)
        content = bundle.tokenizer.decode(ids[:-1])
        messages = [{"role": "user", "content": content + suffix}]
        prompt = bundle.chat_template.render(messages, enable_thinking=False)
    return messages, prompt


async def stream_request(client, profile, messages, *, phase, group, index, record=None,
                         principal_token=None):
    started = time.perf_counter()
    record = record or {}
    consumer = record.get("consumer", {})
    paused = False
    headers = {}
    if record:
        headers["X-Qualification-Request-ID"] = f"{group}/{record['request_id']}"
        if principal_token:
            headers["Authorization"] = "Bearer " + principal_token
    arrivals = []
    metrics = None
    finished = False
    expected_cancel = False
    done = False
    text_parts = []
    try:
        async with client.stream("POST", "/v1/chat/completions", headers=headers, json={
            "model": profile.model_id, "messages": messages, "temperature": 0,
            "seed": profile.seed, "max_tokens": record.get("output_tokens", profile.output_tokens),
            "stream": True,
        }) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                if (not paused and consumer.get("pause_ms")
                        and len(arrivals) >= consumer.get("pause_after_events", 1)):
                    paused = True
                    await asyncio.sleep(consumer["pause_ms"] / 1000)
                if consumer.get("read_delay_ms"):
                    await asyncio.sleep(consumer["read_delay_ms"] / 1000)
                event = json.loads(line[6:])
                if "error" in event:
                    raise BaselineError(json.dumps(event["error"]))
                choice = event.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    text_parts.append(delta.get("content", "") + delta.get("reasoning_content", ""))
                    arrivals.append(time.perf_counter())
                    if ((record.get("cancel_after_events") is not None
                         and len(arrivals) >= record["cancel_after_events"])
                            or (profile.workload == "cancellation" and not record
                                and index == 0 and len(arrivals) == 3)):
                        expected_cancel = True
                        break
                if "mlx" in event:
                    metrics = event["mlx"]
                finished |= choice.get("finish_reason") is not None
        if expected_cancel:
            return {"status": "cancelled_expected", "temperature_phase": phase, "group_id": group,
                    "request_index": index, "started": started, "ended": time.perf_counter()}
        if not finished or not done or metrics is None:
            raise BaselineError("stream omitted terminal metrics or completion marker")
        model_times = metrics.get("model_token_times_ms")
        if model_times is None or metrics.get("execution_observer") != "synchronized-forward-v1":
            raise BaselineError("worker did not supply model-token observations")
        ended = time.perf_counter()
        return {
            "status": "completed", "temperature_phase": phase, "group_id": group,
            "request_index": index, "started": started, "ended": ended,
            "client_ttft_ms": (arrivals[0] - started) * 1000 if arrivals else None,
            "client_latency_ms": (ended - started) * 1000,
            "sse_delivery_gaps_ms": [
                (b - a) * 1000 for a, b in zip(arrivals, arrivals[1:], strict=False)
            ],
            "model_token_gaps_ms": [
                b - a for a, b in zip(model_times, model_times[1:], strict=False)
            ],
            "generated_tokens": metrics["generated_tokens"], "metrics": metrics,
            "output_sha256": hashlib.sha256("".join(text_parts).encode()).hexdigest(),
        }
    except Exception as exc:
        return {"status": "failed", "temperature_phase": phase, "group_id": group,
                "request_index": index, "error": f"{type(exc).__name__}: {exc}",
                "started": started, "ended": time.perf_counter()}


async def replay_group(client, bundle, engine, profile, replay, *, group, phase, token_metadata):
    from mlx_one.inference_qualification_worker import prefix_messages

    started = time.monotonic()
    records = replay["requests"][:1] if phase == "warmup" else replay["requests"]
    async def request(record, index):
        await asyncio.sleep(max(0, started + record["arrival_ms"] / 1000 - time.monotonic()))
        if record.get("messages"):
            messages = record["messages"]
            prompt = bundle.chat_template.render(messages, enable_thinking=False)
        else:
            messages, prompt = prefix_messages(bundle, record["prompt_tokens"],
                record.get("shared_prefix_fraction", 0), "stable", index)
        ids = bundle.tokenizer.encode(prompt)
        if len(ids) + record["output_tokens"] > bundle.context_length:
            raise BaselineError("replay prompt plus output exceeds checkpoint context")
        token_metadata.append({"group": group, "index": index,
            "request_id": record["request_id"], "actual_rendered_tokens": len(ids),
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        tokens = getattr(engine, "_qualification_principal_tokens", {})
        operation = stream_request(client, profile, messages, phase=phase, group=group,
            index=index, record=record, principal_token=tokens.get(record.get("principal_label")))
        limits = [(record[name], name) for name in ("deadline_ms", "cancel_after_ms")
                  if record.get(name) is not None]
        try:
            if limits:
                duration, action = min(limits)
                row = await asyncio.wait_for(operation, duration / 1000)
            else:
                row = await operation
        except asyncio.TimeoutError:
            status = "deadline_expected" if action == "deadline_ms" else "cancelled_expected"
            row = {"status": status,
                   "temperature_phase": phase, "group_id": group, "request_index": index,
                   "started": started, "ended": time.monotonic()}
        return {**row, "replay_request_id": record["request_id"],
                "arrival_ms": record["arrival_ms"],
                "principal_label": record.get("principal_label"),
                "consumer": record.get("consumer", {})}
    return await asyncio.gather(*(request(record, index) for index, record in enumerate(records)))


async def traffic(url, bundle, engine, scheduler, manifest, profile, output):
    import httpx

    rows = []
    config = manifest["profile"]
    token_metadata = []
    extra = {}
    async with httpx.AsyncClient(base_url=url, timeout=profile.timeout_seconds,
                                 headers={"Authorization": "Bearer qualification-local"}) as client:
        def groups():
            yield "cold", "cold"
            yield "warmup", "warmup"
            started = time.monotonic()
            index = 0
            while index < profile.repeats or (
                    profile.duration_seconds is not None
                    and time.monotonic() - started < profile.duration_seconds):
                yield f"warm-{index}", "warm"
                index += 1
            if profile.duration_seconds is not None:
                extra["soak_elapsed_seconds"] = time.monotonic() - started

        async def soak_probe(name):
            messages, _ = build_messages(bundle, profile.context_tokens,
                                         suffix="\nStable recovery probe.")
            result = await stream_request(client, profile, messages,
                                          phase="probe", group=name, index=-1)
            def reconcile():
                import gc

                import mlx.core as mx

                runtime = engine._cache_runtime
                runtime.clear_prefixes()
                runtime.block_pool.discard_free()
                gc.collect()
                mx.clear_cache()
                return {"active_bytes": int(mx.get_active_memory()), "cache": engine.cache_stats()}
            memory = scheduler.execute(reconcile)
            return {"request": result, "memory": memory}

        initial_probe = await soak_probe("initial") if profile.duration_seconds else None
        if profile.workload in {"pressure", "soak"}:
            from mlx_one.inference_qualification_worker import pressure_recovery

            recovery = await pressure_recovery(client, bundle, engine, scheduler, profile, output)
            extra["pressure_recovery_passed"] = recovery["status"] == "passed"
        replay = getattr(engine, "_qualification_replay", None)
        for group, phase in groups():
            if replay is not None:
                group_rows = await replay_group(client, bundle, engine, profile, replay,
                                                 group=group, phase=phase,
                                                 token_metadata=token_metadata)
                rows.extend(group_rows)
                with (output / "requests.jsonl").open("a") as handle:
                    for row in group_rows:
                        handle.write(json.dumps(row) + "\n")
                runtime = (await client.get("/v1/runtime")).json()
                cache = runtime.get("cache") or {}
                extra["peak_physical_bytes"] = max(extra.get("peak_physical_bytes", 0),
                                                    cache.get("physical_bytes") or 0)
                with (output / "time-series.jsonl").open("a") as handle:
                    handle.write(json.dumps({"group": group, "at": time.monotonic(),
                        "cache": cache, "scheduler": runtime.get("scheduler"),
                        "pressure": runtime.get("inference", {}).get("memory_pressure")}) + "\n")
                if any(r["status"] not in {"completed", "cancelled_expected", "deadline_expected"}
                       for r in group_rows):
                    break
                continue
            requests = []
            fanout = (profile.slots * 3 if profile.workload in {
                "turnover", "mixed", "cancellation", "soak", "pressure", "multi_turn"}
                      else profile.slots)
            for index in range(1 if phase == "warmup" else fanout):
                suffix = f"\nReturn the project ID. Request {group}-{index}."
                context = (2048 if profile.workload == "mixed" and index % 2 == 0
                           else profile.context_tokens)
                messages, prompt = build_messages(bundle, context, suffix=suffix)
                if profile.workload == "multi_turn":
                    from mlx_one.inference_qualification_worker import prefix_messages

                    messages, prompt = prefix_messages(bundle, context,
                                                       profile.shared_prefix_fraction,
                                                       group, index)
                prompt_ids = bundle.tokenizer.encode(prompt)
                token_metadata.append({"group": group, "index": index,
                                       "actual_rendered_tokens": len(prompt_ids),
                                       "rendered_prompt_sha256": hashlib.sha256(
                                           prompt.encode()).hexdigest(),
                                       "token_ids_sha256": hashlib.sha256(
                                           json.dumps(prompt_ids).encode()).hexdigest()})
                async def delayed(messages=messages, index=index, phase=phase, group=group):
                    if profile.workload in {"turnover", "mixed", "cancellation"}:
                        await asyncio.sleep(index * .02)
                    return await stream_request(client, profile, messages, phase=phase,
                                                group=group, index=index)
                requests.append(delayed())
            group_rows = await asyncio.gather(*requests)
            rows.extend(group_rows)
            with (output / "requests.jsonl").open("a") as handle:
                for row in group_rows:
                    handle.write(json.dumps(row) + "\n")
            if any(row["status"] not in {"completed", "cancelled_expected"} for row in group_rows):
                break
            runtime = (await client.get("/v1/runtime")).json()
            report = runtime["inference"]
            # Fill provenance unavailable to a general-purpose HTTP runtime reporter.
            report["scope"]["hardware"] = json.dumps(manifest["hardware"], sort_keys=True)
            report["scope"]["source_revision"] = manifest["source"]["source_sha256"]
            write_json(output / "runtime.json", runtime)
            validate_identity(profile, runtime)
            with (output / "time-series.jsonl").open("a") as handle:
                handle.write(json.dumps({"group": group, "at": time.monotonic(),
                                         "cache": runtime.get("cache"),
                                         "pressure": report.get("memory_pressure"),
                                         "scheduler": runtime.get("scheduler")}) + "\n")
        if initial_probe is not None:
            final_probe = await soak_probe("final")
            write_json(output / "soak-probes.json", {"initial": initial_probe,
                                                    "final": final_probe})
            extra.update({"soak_initial_active_bytes": initial_probe["memory"]["active_bytes"],
                          "soak_final_active_bytes": final_probe["memory"]["active_bytes"],
                          "recovery_output_matches": initial_probe["request"].get("output_sha256")
                          == final_probe["request"].get("output_sha256")
                          and final_probe["request"]["status"] == "completed"})
            write_json(output / "memory-reconciliation.json", extra)
        token_metadata.sort(key=lambda row: (row["group"], row["index"]))
        write_json(output / "tokenization.json", token_metadata)
        config["rendered_prompts"] = token_metadata
        write_json(output / "workload.json", config)
    engine._qualification_traffic_stats = extra
    return rows


def execute(manifest):
    import mlx.core as mx
    import uvicorn

    from mlx_one.engine.modes import ACTOR_MODES
    from mlx_one.engine.parity import forced_token_parity
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.generation_engine import GenerationEngine
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.server.scheduler import GenerationScheduler

    output = Path(manifest["output"])
    accepted = {field.name for field in fields(BaselineProfile)}
    profile = BaselineProfile(**{key: value for key, value in manifest["profile"].items()
                                 if key in accepted})
    if profile.inference_mode in ACTOR_MODES:
        from mlx_one.engine.actor import InferenceActor
        from mlx_one.engine.batching import BatchingConfig
        from mlx_one.engine.dispatch import DispatchRunner

        scheduler = InferenceActor(BatchingConfig(
            max_sequences=profile.slots, timeout=profile.timeout_seconds,
            queue_size=max(profile.slots * 4, 8), prefill_chunk_size=profile.chunk_tokens,
            max_scheduled_tokens=profile.max_scheduled_tokens,
            max_prefill_tokens=profile.max_prefill_tokens,
            admission_policy=profile.admission_policy), runner=DispatchRunner())
        scheduler.capture_memory = True
    else:
        scheduler = GenerationScheduler(parallel=profile.slots, timeout=profile.timeout_seconds,
                                        max_pending=max(profile.slots * 4, 8))
    manager = ModelManager()
    engine = GenerationEngine(manager, inference_mode=profile.inference_mode,
                              collect_execution_metrics=True,
                              execution_dtype=profile.execution_dtype,
                              cache_type_k=profile.cache_type, cache_type_v=profile.cache_type,
                              cache_block_size=profile.page_tokens,
                              prefill_chunk_size=profile.chunk_tokens)
    server = None
    thread = None
    listener = None
    try:
        started = time.perf_counter()
        bundle = scheduler.execute(lambda: manager.load(profile.model_id, revision=profile.revision,
                                                        offline=True))
        from mlx_one.engine.modes import configure_execution_dtype

        scheduler.execute(lambda: configure_execution_dtype(bundle, profile.execution_dtype))
        if callable(getattr(scheduler, "set_trace_sink", None)):
            def save_trace(trace):
                with (output / "forward-traces.jsonl").open("a") as handle:
                    handle.write(json.dumps(trace) + "\n")
            scheduler.set_trace_sink(save_trace)
        if callable(getattr(scheduler, "set_event_sink", None)):
            event_lock = threading.Lock()
            def save_event(event):
                with event_lock, (output / "actor-events.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
            scheduler.set_event_sink(save_event)
        write_json(output / "load.json", {"load_seconds": time.perf_counter() - started,
                                          "mlx_peak_bytes": int(mx.get_peak_memory())})
        # Hash exact files, including tokenizer/config, without changing the cache.
        files = {}
        for path in sorted(bundle.path.iterdir()):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".txt", ".jinja"}:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[path.name] = digest.hexdigest()
        write_json(output / "checkpoint.json", {"model_id": bundle.model_id,
                                                "revision": bundle.revision, "files": files,
                                                "quantization": bundle.quantization})
        if profile.inference_mode in ACTOR_MODES:
            engine.bind_actor(scheduler)
        runtime_owner = scheduler.execute(lambda: engine._cache_manager_for(bundle))
        runtime_owner.harness_only = True
        if profile.workload == "microbench":
            from mlx_one.inference_qualification_worker import attention_microbench

            measured = scheduler.execute(lambda: attention_microbench(bundle, profile))
            write_json(output / "microbench.json", measured)
            if not profile.kv_bits and not measured["within_tolerance"]:
                raise BaselineError("direct attention microbenchmark parity failed")
            return
        if profile.workload == "quality":
            from mlx_one.inference_qualification_worker import quality_run

            scheduler.execute(lambda: quality_run(bundle, engine, profile, output))
            return
        if profile.workload == "parity":
            _, prompt = build_messages(bundle, profile.context_tokens,
                                       suffix="\nContinue deterministically.")
            if profile.inference_mode != "reference_dense":
                from mlx_one.engine.qualification import execution_parity

                result = scheduler.execute(lambda: execution_parity(
                    bundle, mode=profile.inference_mode, prompt=prompt, count=profile.output_tokens,
                    batch_size=profile.slots, chunk_size=profile.chunk_tokens,
                    cache_type=profile.cache_type, page_tokens=profile.page_tokens,
                    atol=profile.atol, rtol=profile.rtol, seed=profile.seed))
                report = result["runtime"]
                report["scope"]["hardware"] = json.dumps(manifest["hardware"], sort_keys=True)
                report["scope"]["source_revision"] = manifest["source"]["source_sha256"]
                validate_identity(profile, {"inference": report})
                write_json(output / "parity.json", result)
                write_json(output / "cleanup.json", result["cleanup"])
                feature = {"experimental_block": "block_prefix_reuse", "paged": "paged_attention",
                           "experimental_hybrid_batch": "hybrid_prefix_restore"}.get(
                               profile.inference_mode,
                               "batched_dense_decode" if profile.slots > 1 else "dense_execution")
                qualification_evidence(output, feature=feature,
                    scope=report["scope"], parity=result)
                if result["status"] == "failed":
                    raise BaselineError("optimized execution parity failed")
                return
            result = scheduler.execute(lambda: forced_token_parity(
                bundle, prompt=prompt, count=profile.output_tokens,
                atol=profile.atol, rtol=profile.rtol, seed=profile.seed,
                prefill_chunk_tokens=512))
            write_json(output / "parity.json", result)
            if result["status"] != "passed":
                raise BaselineError("forced-token parity failed")
            return
        config = ServerConfig(
            inference_mode=profile.inference_mode, warmup=False, parallel=profile.slots,
            timeout=profile.timeout_seconds, reasoning="off",
            api_keys=("qualification-local",), queue_size=max(profile.slots * 4, 8),
            execution_dtype=profile.execution_dtype, prefill_chunk_size=profile.chunk_tokens,
            max_batch_tokens=profile.max_scheduled_tokens,
            max_prefill_tokens=profile.max_prefill_tokens,
            admission_policy=profile.admission_policy,
            cache_block_size=profile.page_tokens,
            cache_prompt=profile.workload in {"prefix", "multi_turn"},
            cache_idle_slots=profile.workload in {"prefix", "multi_turn"},
            cache_type_k=profile.cache_type, cache_type_v=profile.cache_type,
        )
        from mlx_one.traffic_replay import (
            QualificationMiddleware,
            generated_traffic,
            load_traffic_manifest,
        )

        replay = None
        if profile.traffic_manifest:
            loaded = load_traffic_manifest(profile.traffic_manifest)
            if loaded["sha256"] != profile.traffic_identity:
                raise BaselineError("traffic manifest changed after profile creation")
            replay = loaded["manifest"]
        elif profile.workload in {"mixed", "turnover", "cancellation", "slow_consumer",
                                  "multi_turn", "soak"}:
            replay = generated_traffic(profile)
        if replay:
            import secrets
            from dataclasses import replace

            labels = {r.get("principal_label") for r in replay["requests"]
                      if r.get("principal_label") is not None}
            if labels and any(r.get("principal_label") is None for r in replay["requests"]):
                raise BaselineError(
                    "authenticated and anonymous traffic require separate server runs")
            engine._qualification_principal_tokens = {label: secrets.token_hex(32)
                                                       for label in labels}
            engine._qualification_replay = replay
            config = replace(config, api_keys=("qualification-local", *(
                engine._qualification_principal_tokens.values())) if labels else ())
            write_json(output / "traffic.json", replay)
        app = create_app(manager, engine, scheduler=scheduler, config=config)
        if replay:
            app = QualificationMiddleware(app, replay)
        # Bind before starting uvicorn to avoid a free-port selection race.
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            app, log_level="error",
            lifespan="on",
        ))
        thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise BaselineError("HTTP worker failed to start")
            time.sleep(.01)
        rows = asyncio.run(traffic(f"http://127.0.0.1:{port}", bundle, engine, scheduler,
                                   manifest, profile, output))
        scheduler.execute(lambda: None)  # Drain cancellation and disposal before cleanup evidence.
        summary = summarize(rows)
        summary.update(getattr(engine, "_qualification_traffic_stats", {}))
        summary["peak_cache_bytes"] = max((r.get("metrics", {}).get("cache_bytes", 0)
                                           for r in rows), default=0)
        from mlx_one.inference_qualification import qualified_active_decoders

        summary["declared_decode_floor_tok_s"] = profile.min_decode_tokens_per_second
        summary["qualified_active_decoders"] = qualified_active_decoders(rows, profile)
        summary["expected_cancellations"] = sum(r["status"] == "cancelled_expected" for r in rows)
        summary["observed_cancellations"] = scheduler.stats().get("cancelled_requests", 0)
        runtime = json.loads((output / "runtime.json").read_text())
        scope = runtime["inference"]["scope"]
        summary["identity"] = {"hardware": manifest["hardware"], "source": manifest["source"],
                               **{key: scope[key] for key in
                                  ("weight_precision", "cache_precision", "mlx_version")}}
        summary["profile"] = profile.to_dict()
        from mlx_one.traffic_replay import progress_fairness

        event_path = output / "actor-events.jsonl"
        events = [json.loads(line) for line in event_path.read_text().splitlines()] if (
            event_path.exists()) else []
        fairness = progress_fairness(events, max_queue_wait_ms=profile.max_queue_wait_ms,
                                     max_service_gap_ms=profile.max_service_gap_ms)
        write_json(output / "fairness.json", fairness)
        summary["fairness"] = fairness
        summary["starved_requests"] = fairness["starved_requests"]
        from mlx_one.inference_baseline import percentile

        summary["queue_wait_p95_ms"] = percentile([r["metrics"]["queue_ms"] for r in rows
            if r.get("metrics", {}).get("queue_ms") is not None], .95)
        rates = summary.get("aggregate_output_tokens_per_second_by_group", [])
        summary["warm_aggregate_output_tokens_per_second"] = (
            sum(rates) / len(rates) if rates and all(r is not None for r in rates) else None)
        summary["prefix_hits"] = engine.cache_stats().get("prefix_hits")
        summary["prefix_tokens_reused"] = engine.cache_stats().get("prefix_tokens_reused")
        write_json(output / "summary.json", summary)
        if profile.inference_mode == "experimental_dense_batch":
            from mlx_one.engine.qualification import evaluate_gate_c, execution_parity

            parity = scheduler.execute(lambda: execution_parity(
                bundle, mode=profile.inference_mode, count=32, batch_size=profile.slots,
                chunk_size=profile.chunk_tokens, atol=profile.atol, rtol=profile.rtol))
            write_json(output / "parity.json", parity)
            reference = None
            if profile.reference_run:
                directory = Path(profile.reference_run)
                reference_summary = json.loads((directory / "summary.json").read_text())
                reference = {"summary": reference_summary,
                             "profile": reference_summary["profile"],
                             "identity": reference_summary["identity"]}
            traces = [json.loads(line) for line in
                      (output / "forward-traces.jsonl").read_text().splitlines()]
            summary["forward_count"] = len(traces)
            summary["decode_forward_count"] = sum(t["phase"] == "decode" for t in traces)
            summary["real_scheduled_tokens"] = sum(t["real_tokens"] for t in traces)
            summary["tokens_per_forward"] = (summary["real_scheduled_tokens"] / len(traces)
                                               if traces else None)
            write_json(output / "summary.json", summary)
            gate = evaluate_gate_c(profile, summary, traces, parity, reference)
            write_json(output / "gate-c.json", gate)
            if gate["status"] == "failed":
                raise BaselineError("declared Gate C acceptance failed")
            if parity["status"] != "passed":
                raise BaselineError("batched logits parity failed")
        with (output / "requests.csv").open("w", newline="") as handle:
            columns = ("status", "temperature_phase", "group_id", "request_index",
                       "client_ttft_ms", "client_latency_ms", "generated_tokens", "error")
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        if summary["failed"]:
            raise BaselineError("one or more baseline requests failed")
        if profile.inference_mode not in ACTOR_MODES and any(
            size != 1 for row in rows for size in row.get("metrics", {}).get(
            "model_forward_batch_sizes", [])):
            raise BaselineError("cooperative baseline unexpectedly executed B>1")
        cleanup = engine.cache_stats()
        write_json(output / "cleanup.json", cleanup)
        if cleanup.get("active_handles") or cleanup.get("reserved_bytes"):
            raise BaselineError("baseline leaked live handles or reservations")
        if profile.inference_mode in {"paged", "experimental_hybrid_batch"}:
            from mlx_one.inference_qualification_worker import advanced_checks

            parity, gates, scope, metadata = advanced_checks(
                bundle, engine, scheduler, profile, summary, output)
            feature = ("paged_attention" if profile.inference_mode == "paged"
                       else "hybrid_prefix_restore")
            qualification_evidence(output, feature=feature, scope=scope, parity=parity,
                                   gate=gates.get("d", gates.get("hybrid-performance")), **metadata)
            if profile.kv_bits:
                qualification_evidence(output, feature="quantized_paged_kv", scope=scope,
                                       parity=parity, gate=gates.get("d"), **metadata)
            if (profile.context_tokens + profile.output_tokens >= 128000
                    and metadata["quality_passed"] and metadata["memory_passed"]
                    and metadata["service_passed"]):
                qualification_evidence(output, feature="long_context", scope=scope,
                                       parity=parity, **metadata)
        if profile.inference_mode == "experimental_dense_batch":
            feature = "batched_dense_decode" if profile.slots > 1 else "dense_execution"
            qualification_evidence(output, feature=feature, scope=scope, parity=parity, gate=gate)
            if gate["status"] == "passed":
                qualification_evidence(output, feature="continuous_batching", scope=scope,
                                       parity=parity, gate=gate)
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)
        if listener is not None:
            listener.close()
        scheduler.close(finalizer=lambda: (engine.close(), manager.unload()))
        from mlx_one.inference_qualification import write_integrity

        write_integrity(output)


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    output = Path(manifest["output"])
    try:
        execute(manifest)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                              "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
