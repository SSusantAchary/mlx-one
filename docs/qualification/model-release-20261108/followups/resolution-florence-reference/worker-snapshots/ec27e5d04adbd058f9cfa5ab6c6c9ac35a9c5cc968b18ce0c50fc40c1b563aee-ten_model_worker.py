"""Isolated exact-checkpoint worker for native mathematics and public interfaces."""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/model_parity"))


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(reference, candidate, dtype):
    import mlx.core as mx
    import numpy as np

    from mlx_one.ops.numerical_trace import compare_tensors, tensor_metadata

    mx.eval(reference, candidate)
    left, right = (np.asarray(x.astype(mx.float32)) for x in (reference, candidate))
    result = compare_tensors(
        left,
        right,
        dtype=dtype,
        reference_metadata=tensor_metadata(reference),
        candidate_metadata=tensor_metadata(candidate),
    )
    if left.shape == right.shape:
        difference = left.astype(np.float64) - right
        result["rms_error"] = float(np.sqrt(np.mean(difference**2)))
        denominator = float(np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel()))
        result["cosine_similarity"] = (
            float(np.dot(left.ravel(), right.ravel()) / denominator) if denominator else None
        )
        result["top1_agreement"] = float(np.mean(left.argmax(-1) == right.argmax(-1)))
        result["top5_set_agreement"] = float(
            np.mean(
                np.sort(np.argpartition(left, -5, axis=-1)[..., -5:], axis=-1)
                == np.sort(np.argpartition(right, -5, axis=-1)[..., -5:], axis=-1)
            )
        )
    return result


def cache_arrays(cache, family, *, reference=False):
    import mlx.core as mx

    if hasattr(cache, "conv_state"):
        return (cache.conv_state, cache.recurrent_state)
    if hasattr(cache, "cache"):
        values = tuple(cache.cache)
        if family == "qwen3_5" and len(values) == 2 and values[1] is not None:
            return (values[0], values[1].swapaxes(-1, -2))
        return values
    if hasattr(cache, "keys"):
        if reference and hasattr(cache, "_temporal_order") and cache.keys is not None:
            return tuple(
                cache._temporal_order(x)[:, :, -cache.max_size :]
                for x in (cache.keys, cache.values)
            )
        if reference and hasattr(cache, "keys_and_values"):
            return cache.keys_and_values()
        state = cache.state() if callable(cache.state) else cache.state
        return tuple(x for x in state if isinstance(x, mx.array))
    return (getattr(cache, "values", None),)


def numerical(bundle, reference, gate):
    import mlx.core as mx
    from layerwise import capture_modules, compare_events
    from local_reference import reference_module

    seed = bundle.tokenizer.encode("The capital of France is")
    prompt = (seed * ((16 + len(seed) - 1) // len(seed)))[:16]
    tokens = mx.array([prompt])
    family = bundle.architecture
    with capture_modules(reference, family) as ref_events:
        reference_full = reference(tokens)
    with capture_modules(bundle.model, family) as native_events:
        native_full = bundle.model(tokens).logits
    trace = compare_events(ref_events, native_events, gate)
    nc, rc = bundle.model.make_cache(), reference_module("cache").make_prompt_cache(reference)
    native_parts = [bundle.model(tokens, cache=nc).logits]
    reference_parts = [reference(tokens, cache=rc)]
    boundaries = [metrics(reference_full, native_full, gate)]
    native_ids, ref_ids = [], []
    decode_trace = None
    for index in range(16):
        native_ids.append(int(mx.argmax(native_parts[-1][0, -1]).item()))
        ref_ids.append(int(mx.argmax(reference_parts[-1][0, -1]).item()))
        token = mx.array([[ref_ids[-1]]])
        if index == 7:
            with capture_modules(reference, family) as ref_events:
                ref_out = reference(token, cache=rc)
            with capture_modules(bundle.model, family) as native_events:
                native_out = bundle.model(token, cache=nc).logits
            decode_trace = compare_events(ref_events, native_events, gate)
        else:
            ref_out, native_out = reference(token, cache=rc), bundle.model(token, cache=nc).logits
        native_parts.append(native_out)
        reference_parts.append(ref_out)
        boundaries.append(metrics(ref_out, native_out, gate))
    continued = mx.array([prompt + ref_ids])
    native_continued = bundle.model(continued).logits
    reference_continued = reference(continued)
    native_state = [cache_arrays(c, family) for c in nc]
    reference_state = [cache_arrays(c, family, reference=True) for c in rc]
    state_metrics = []
    for index, (left, right) in enumerate(zip(reference_state, native_state, strict=True)):
        for component, (ref, got) in enumerate(zip(left, right, strict=True)):
            if ref is None or got is None:
                continue
            state_metrics.append(
                {
                    "layer": index,
                    "component": component,
                    **metrics(ref, got, str(ref.dtype).rsplit(".", 1)[-1]),
                }
            )
    own = {
        "native_full_vs_decode": metrics(
            native_continued, mx.concatenate(native_parts, axis=1), gate
        ),
        "reference_full_vs_decode": metrics(
            reference_continued, mx.concatenate(reference_parts, axis=1), gate
        ),
    }
    cc = bundle.model.make_cache()
    chunked = mx.concatenate(
        [bundle.model(tokens[:, a:b], cache=cc).logits for a, b in [(0, 3), (3, 11), (11, 16)]],
        axis=1,
    )
    own["native_full_vs_chunked"] = metrics(native_full, chunked, gate)
    rcc = reference_module("cache").make_prompt_cache(reference)
    reference_chunked = mx.concatenate(
        [reference(tokens[:, a:b], cache=rcc) for a, b in [(0, 3), (3, 11), (11, 16)]],
        axis=1,
    )
    own["reference_full_vs_chunked"] = metrics(reference_full, reference_chunked, gate)
    for cache in cc:
        cache.reset()
    own["native_reset"] = metrics(native_full, bundle.model(tokens, cache=cc).logits, gate)
    sequences = []
    for model, native in ((reference, False), (bundle.model, True)):
        cache = model.make_cache() if native else reference_module("cache").make_prompt_cache(model)
        output = model(tokens, cache=cache)
        ids, finish = [], "length"
        for _ in range(16):
            logits = output.logits if native else output
            token = int(mx.argmax(logits[0, -1]).item())
            if token in bundle.eos_token_ids:
                finish = "stop"
                break
            ids.append(token)
            output = model(mx.array([[token]]), cache=cache)
        sequences.append(
            {"token_ids": ids, "text": bundle.tokenizer.decode(ids), "finish_reason": finish}
        )
    passed = all(m["status"] == "PASS" and m["dtype_match"] for m in boundaries)
    passed &= all(m["status"] == "PASS" for m in [*own.values(), *state_metrics])
    return {
        "status": "PASS" if passed else "FAIL_NUMERICAL",
        "gate_dtype": gate,
        "logits": boundaries,
        "state": state_metrics,
        "continuation": own,
        "generated_token_parity": native_ids == ref_ids,
        "native_ids_on_reference_continuation": native_ids,
        "reference_ids": ref_ids,
        "independent_greedy": {
            "reference": sequences[0],
            "native": sequences[1],
            "token_parity": sequences[0]["token_ids"] == sequences[1]["token_ids"],
            "text_parity": sequences[0]["text"] == sequences[1]["text"],
            "finish_parity": sequences[0]["finish_reason"] == sequences[1]["finish_reason"],
        },
        "continuation_contract": (
            "Reference token IDs are forced into both models; token agreement is "
            "teacher-forced, not independent greedy sequence parity"
        ),
        "prefill_layerwise": trace,
        "decode_step_7_layerwise": decode_trace,
        "native_state_dtypes": [
            [str(x.dtype) for x in values if x is not None] for values in native_state
        ],
        "recurrent_layout_mapping": "reference [B,H,V,K] transposed to native [B,H,K,V]",
    }


def generation_and_python(bundle, execution_dtype):
    from mlx_one import Engine
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.text import TextGenerationOptions

    prompt = "The capital of France is"
    options = TextGenerationOptions(max_tokens=16, temperature=0)
    with Engine(
        bundle,
        execution_dtype=execution_dtype,
        inference_mode="reference_dense",
        cache_config=CacheConfig(prefix_cache_enabled=False),
    ) as engine:
        first, second = (
            engine.generate(prompt, options=options),
            engine.generate(prompt, options=options),
        )
        chunks = list(engine.stream(prompt, options=options))
        ids = [x.token_id for x in chunks if x.token_id is not None]
        repeated_ids = [
            x.token_id for x in engine.stream(prompt, options=options) if x.token_id is not None
        ]
        stream_text = "".join(x.text for x in chunks)
        checks = {
            "repeat": first.text == second.text,
            "deterministic_token_ids": ids == repeated_ids,
            "stream_text": first.text == stream_text,
            "finish_reason": chunks[-1].finish_reason == first.finish_reason,
            "usage": len(ids) == first.generation_tokens <= 16,
            "cleanup": engine.runtime_stats()["cache_stats"]["active_handles"] == 0,
        }
        stream = engine.stream(prompt, options=TextGenerationOptions(max_tokens=64, temperature=0))
        next(stream)
        stream.close()
        checks["cancellation_cleanup"] = (
            engine.runtime_stats()["cache_stats"]["active_handles"] == 0
        )
        after = engine.generate(prompt, options=options)
        checks["post_cancel_restart"] = after.text == first.text
        sampled_options = replace(options, max_tokens=8, temperature=0.7, top_p=0.9, seed=37)
        sampled = [list(engine.stream(prompt, options=sampled_options)) for _ in range(2)]
        checks["seeded_sampling"] = [c.token_id for c in sampled[0]] == [
            c.token_id for c in sampled[1]
        ]
        # Stop a matched visible prefix; do not invent a stop token absent from output.
        if first.text:
            stop = first.text[: min(4, len(first.text))]
            stopped = engine.generate(prompt, options=replace(options, stop=(stop,)))
            checks["stop"] = stopped.finish_reason == "stop" and stop not in stopped.text
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL_BEHAVIOR",
            "checks": checks,
            "prompt": prompt,
            "text": first.text,
            "token_ids": ids,
            "finish_reason": first.finish_reason,
            "usage": first.generation_tokens,
            "capabilities": engine.capabilities(),
        }
    return report


def cli(bundle, expected, *, compatible=True):
    if not compatible:
        return {"status": "UNSUPPORTED", "reason": "CLI has no in-memory precision override"}
    process = subprocess.run(
        [
            str(ROOT / ".venv/bin/mlx-one"),
            "generate",
            str(bundle.path),
            expected["prompt"],
            "--offline",
            "--max-tokens",
            "16",
            "--temperature",
            "0",
            "--stream",
            "--json-output",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if process.returncode:
        return {
            "status": "FAIL_API",
            "exit_code": process.returncode,
            "stderr": process.stderr[-3000:],
        }
    chunks = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    ids = [x["token_id"] for x in chunks if x["token_id"] is not None]
    text = "".join(x["text"] for x in chunks)
    passed = ids == expected["token_ids"] and text == expected["text"]
    return {
        "status": "PASS" if passed else "FAIL_API",
        "text": text,
        "ids": ids,
        "finish_reason": chunks[-1]["finish_reason"],
        "stderr": process.stderr[-1000:],
    }


def http_checks(manager, bundle, expected, execution_dtype, chat_expected):
    import httpx
    import uvicorn

    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.generation_engine import GenerationEngine
    from mlx_one.text import TextGenerationOptions, stream_chat

    engine = GenerationEngine(
        manager,
        execution_dtype=execution_dtype,
        inference_mode="reference_dense",
        collect_execution_metrics=True,
    )
    app = create_app(
        manager,
        engine,
        config=ServerConfig(
            warmup=False,
            cache_prompt=False,
            inference_mode="reference_dense",
            execution_dtype=execution_dtype,
            reasoning="off",
        ),
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    checks, responses = {}, {}
    try:
        deadline = time.monotonic() + 15
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("socket server startup failed")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
            checks["health"] = client.get("/health").status_code == 200
            models = client.get("/v1/models").json()["data"]
            checks["models"] = len(models) == 1 and models[0]["id"] == bundle.model_id
            checks["runtime"] = client.get("/v1/runtime").status_code == 200
            payload = dict(
                model=bundle.model_id,
                prompt=expected["prompt"],
                max_tokens=16,
                temperature=0,
                top_p=1.0,
            )
            request_started = time.perf_counter()
            result = client.post("/v1/completions", json=payload)
            responses["completion_wall_ms"] = (time.perf_counter() - request_started) * 1000
            result.raise_for_status()
            responses["completion"] = result.json()
            checks["completion"] = result.json()["choices"][0]["text"] == expected["text"]
            checks["finish"] = (
                result.json()["choices"][0]["finish_reason"] == expected["finish_reason"]
            )
            checks["usage"] = result.json()["usage"]["completion_tokens"] == expected["usage"]
            checks["raw_sse"] = sse_check(client, "/v1/completions", payload, expected["text"])
            checks["unknown_model"] = (
                client.post("/v1/completions", json={**payload, "model": "missing"}).status_code
                == 404
            )
            checks["malformed"] = (
                client.post("/v1/completions", json={**payload, "prompt": []}).status_code == 422
            )
            for label, replacement in (
                ("invalid_temperature", {"temperature": -1}),
                ("invalid_top_p", {"top_p": 0}),
                ("negative_tokens", {"max_tokens": -1}),
                ("unsupported_precision", {"execution_dtype": "float128"}),
            ):
                invalid = client.post("/v1/completions", json={**payload, **replacement})
                checks[label] = invalid.status_code == 422 and isinstance(invalid.json(), dict)
            for label, messages in (
                ("missing_messages", None),
                ("invalid_role", [{"role": "invalid", "content": "hello"}]),
            ):
                body = {"model": bundle.model_id, "max_tokens": 1}
                if messages is not None:
                    body["messages"] = messages
                invalid = client.post("/v1/chat/completions", json=body)
                checks[label] = invalid.status_code == 422 and isinstance(invalid.json(), dict)
            invalid = client.post("/v1/embeddings", json={"model": bundle.model_id, "input": "hi"})
            checks["unsupported_task"] = invalid.status_code == 400 and "error" in invalid.json()
            checks["overflow"] = (
                client.post(
                    "/v1/completions", json={**payload, "prompt": " a" * bundle.context_length}
                ).status_code
                == 400
            )
            checks["cleanup"] = engine.cache_stats()["active_handles"] == 0
            if chat_expected and bundle.chat_template and bundle.chat_template.template:
                messages = [{"role": "user", "content": "Reply only YES."}]
                chunks = manager._pool.executor.call(
                    bundle.model_id,
                    lambda: list(
                        stream_chat(
                            bundle,
                            messages,
                            options=TextGenerationOptions(max_tokens=16, temperature=0),
                            template_options={"enable_thinking": False},
                        )
                    ),
                )
                chat_text = "".join(c.text for c in chunks)
                body = dict(
                    model=bundle.model_id,
                    messages=messages,
                    max_tokens=16,
                    temperature=0,
                    top_p=1.0,
                )
                result = client.post("/v1/chat/completions", json=body)
                result.raise_for_status()
                responses["chat"] = result.json()
                checks["chat"] = result.json()["choices"][0]["message"]["content"] == chat_text
                checks["chat_sse"] = sse_check(client, "/v1/chat/completions", body, chat_text)
            else:
                responses["chat"] = {
                    "status": "UNSUPPORTED",
                    "reason": "no applicable exact template",
                }
            # Close a live wire stream early, then verify a clean subsequent request.
            with client.stream(
                "POST", "/v1/completions", json={**payload, "max_tokens": 128, "stream": True}
            ) as stream:
                for line in stream.iter_lines():
                    if line.startswith("data: "):
                        break
            time.sleep(0.1)
            checks["post_disconnect"] = (
                client.post("/v1/completions", json=payload).status_code == 200
            )
            checks["post_disconnect_cleanup"] = engine.cache_stats()["active_handles"] == 0
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("HTTP server did not stop")
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_API",
        "checks": checks,
        "responses": responses,
        "transport": "loopback TCP",
        "server_shutdown": True,
    }


def sse_check(client, route, body, expected):
    response = client.post(route, json={**body, "stream": True})
    response.raise_for_status()
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    text = "".join(
        c["choices"][0].get("text", c["choices"][0].get("delta", {}).get("content", ""))
        for c in events
        if "choices" in c
    )
    terminals = sum(
        c["choices"][0].get("finish_reason") is not None for c in events if "choices" in c
    )
    return (
        text == expected
        and terminals == 1
        and response.text.endswith("data: [DONE]\n\n")
        and "text/event-stream" in response.headers["content-type"]
        and not any("error" in c for c in events)
    )


def hardware(bundle, execution_dtype, contexts, outputs, progress=None):
    import mlx.core as mx

    from mlx_one import Engine
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.engine.observation import observe_request
    from mlx_one.engine.timing import RequestTiming
    from mlx_one.text import TextGenerationOptions

    rows = []
    with Engine(
        bundle,
        execution_dtype=execution_dtype,
        inference_mode="reference_dense",
        cache_config=CacheConfig(prefix_cache_enabled=False),
    ) as engine:
        engine.generate("Warmup", options=TextGenerationOptions(max_tokens=1))
        for context in contexts:
            prompt = " a" * context
            actual_context = len(bundle.tokenizer.encode(prompt))
            for output in outputs:
                if actual_context + output > bundle.context_length:
                    rows.append(
                        {
                            "input_tokens": actual_context,
                            "output_budget": output,
                            "status": "UNSUPPORTED",
                            "reason": "checkpoint context limit",
                        }
                    )
                    continue
                samples = []
                for _ in range(3):
                    timing = RequestTiming(time.perf_counter(), trace=True)
                    with observe_request(timing.observe):
                        result = engine.generate(
                            prompt, options=TextGenerationOptions(max_tokens=output, temperature=0)
                        )
                    sample = timing.summary()
                    prefill_ms = timing.forward_ms["prefill"]
                    sample.update(
                        prefill_tokens_per_second=actual_context * 1000 / prefill_ms
                        if prefill_ms
                        else None,
                        prefill_ms=prefill_ms,
                        actual_generated_tokens=result.generation_tokens,
                        peak_mlx_bytes=mx.get_peak_memory(),
                        active_handles=engine.runtime_stats()["cache_stats"]["active_handles"],
                    )
                    samples.append(sample)
                rows.append(
                    {
                        "input_tokens": actual_context,
                        "output_budget": output,
                        "status": "PASS"
                        if all(s["active_handles"] == 0 for s in samples)
                        else "FAIL_BEHAVIOR",
                        "samples": samples,
                    }
                )
                if progress:
                    progress(rows)
    return {
        "status": "PASS"
        if all(r["status"] in {"PASS", "UNSUPPORTED"} for r in rows)
        else "FAIL_BEHAVIOR",
        "rows": rows,
        "batch_size": 1,
        "repeats": 3,
        "note": "Only listed context/output cells executed; no performance superiority claim",
    }


def run(request, output):
    import mlx.core as mx
    import mlx.nn as nn
    from local_reference import reference_module

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.reporting import tensor_dtypes
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.text.loading import _prepare_quantized_model, _read_safetensors

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(1024**3)

    report = {**request, "status": "NOT_RUN", "qualification": "NOT_QUALIFIED", "checks": {}}

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    checks = set(request["checks"])
    all_checks = "all" in checks
    precision = request["precision"]
    execution_dtype = (
        None
        if precision == "stored"
        else ("float16" if precision.startswith("affine_") else precision)
    )
    manager = ModelManager()
    stage = "loading"

    def attempt(name, function):
        nonlocal stage
        stage = name
        try:
            report["checks"][name] = function()
        except Exception as exc:
            import traceback

            report["checks"][name] = {
                "status": "FAIL_NUMERICAL" if name == "numerical" else "FAIL_API",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        save()

    try:
        started = time.perf_counter()
        bundle = manager.load(request["path"], offline=True)
        if bundle.model_id != request["repo"] or bundle.revision != request["revision"]:
            raise ValueError("exact checkpoint identity differs from manifest")
        report["cold_load_seconds"] = time.perf_counter() - started
        report["checks"]["weights"] = {"status": "PASS", "strict_loading": True}
        report["checks"]["architecture"] = {"status": "PASS", "family": bundle.architecture}
        path = Path(request["path"])
        report["checkpoint"] = {
            "model_id": bundle.model_id,
            "revision": bundle.revision,
            "context_tokens": bundle.context_length,
            "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "tokenizer_identity": bundle.tokenizer_identity,
            "template_present": bool(bundle.chat_template and bundle.chat_template.template),
            "files": {
                p.name: file_digest(p)
                for p in path.glob("*")
                if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja"}
            },
        }

        def prepare_precision():
            nonlocal bundle
            if precision.startswith("affine_"):
                bits = int(precision[-1])
                nn.quantize(bundle.model, bits=bits, group_size=64, mode="affine")
                bundle = replace(
                    bundle, quantization={"bits": bits, "group_size": 64, "mode": "affine"}
                )
                manager._bundle = bundle
                report["precision_origin"] = (
                    "in-memory MLX affine transform of exact upstream weights"
                )
            configure_execution_dtype(bundle, execution_dtype)
            mx.eval(bundle.model.parameters())
            report["precision_profile"] = {
                "requested": precision,
                "quantization": bundle.quantization,
                "actual_weight_dtypes": tensor_dtypes(bundle.model.parameters()),
                "execution_dtype": execution_dtype,
                "quantized_modules": [
                    {
                        "name": name,
                        "bits": getattr(module, "bits", None),
                        "group_size": getattr(module, "group_size", None),
                    }
                    for name, module in bundle.model.named_modules()
                    if getattr(module, "bits", None) is not None
                ],
                "packed_execution": (
                    "MLX quantized matmul; explicit stable INT8 policy may gather/dequantize "
                    "tiles; no claim of packed-only kernels"
                ),
            }

        stage = "precision"
        manager._pool.executor.call(bundle.model_id, prepare_precision)
        save()
        if all_checks or "numerical" in checks:

            def compute_reference():
                nonlocal bundle
                family = bundle.architecture
                module = reference_module(family)
                config = json.loads((path / "config.json").read_text())
                ref = module.Model(module.ModelArgs.from_dict(config))
                tensors = _read_safetensors(path)
                if hasattr(ref, "sanitize"):
                    tensors = ref.sanitize(tensors)
                if bundle.quantization and not precision.startswith("affine_"):
                    _prepare_quantized_model(ref, tensors, bundle.quantization)
                ref.load_weights(list(tensors.items()), strict=True)
                ref.eval()
                del tensors
                if precision.startswith("affine_"):
                    bits = int(precision[-1])
                    nn.quantize(ref, bits=bits, group_size=64, mode="affine")
                if execution_dtype:
                    ref.set_dtype(
                        getattr(mx, execution_dtype),
                        predicate=lambda dtype: (
                            str(dtype).rsplit(".", 1)[-1] in {"float32", "float16", "bfloat16"}
                        ),
                    )
                mx.eval(bundle.model.parameters(), ref.parameters())
                dtype_set = tensor_dtypes(bundle.model.parameters())
                gate = execution_dtype or next(
                    d for d in dtype_set if d in {"float32", "float16", "bfloat16"}
                )
                report["checks"]["numerical"] = numerical(bundle, ref, gate)
                report["reference"] = {
                    "model_source": module.__file__,
                    "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                    "installed_comparator": False,
                }
                del ref
                gc.collect()
                mx.clear_cache()
                return report["checks"]["numerical"]

            attempt(
                "numerical", lambda: manager._pool.executor.call(bundle.model_id, compute_reference)
            )
        expected = None
        if all_checks or checks & {"generation", "python-api", "cli", "http", "streaming"}:
            attempt("generation", lambda: generation_and_python(bundle, execution_dtype))
            expected = report["checks"]["generation"]
            report["checks"]["python-api"] = {"status": expected["status"]}
            save()
        if all_checks or "cli" in checks:
            attempt("cli", lambda: cli(bundle, expected, compatible=precision == "stored"))
        if all_checks or "quality" in checks:

            def quality():
                from quality import evaluate

                from mlx_one import Engine
                from mlx_one.engine.cache_specs import CacheConfig

                with Engine(
                    bundle,
                    execution_dtype=execution_dtype,
                    inference_mode="reference_dense",
                    cache_config=CacheConfig(prefix_cache_enabled=False),
                ) as engine:
                    return evaluate(
                        bundle,
                        engine,
                        Path(__file__).with_name("ten_model_quality_v1.json"),
                        chat_applicable=request["chat"],
                    )

            attempt("quality", quality)
        if all_checks or "hardware" in checks:

            def progress(rows):
                report["checks"]["hardware"] = {
                    "status": "NOT_RUN",
                    "rows": rows,
                    "reason": "campaign still running",
                }
                save()

            attempt(
                "hardware",
                lambda: hardware(
                    bundle, execution_dtype, request["contexts"], request["outputs"], progress
                ),
            )
        if all_checks or checks & {"http", "streaming"}:
            attempt(
                "http",
                lambda: http_checks(manager, bundle, expected, execution_dtype, request["chat"]),
            )
        failed = [
            v["status"]
            for v in report["checks"].values()
            if v["status"] not in {"PASS", "UNSUPPORTED"}
        ]
        report["status"] = failed[0] if failed else "PASS"
        report["release_gate"] = (
            "NOT_QUALIFIED: scoped checks only; complete context/precision/global gates separate"
        )
    except Exception as exc:
        report["status"] = "FAIL_LOADING" if stage in {"loading", "precision"} else "FAIL_API"
        report["failed_stage"] = stage
        report["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        report["diagnostic_traceback"] = traceback.format_exc()
    finally:
        manager.unload()
        report["peak_mlx_bytes"] = mx.get_peak_memory()
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["hardware"] = mx.device_info()
        save()


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
