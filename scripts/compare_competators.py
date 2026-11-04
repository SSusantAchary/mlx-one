"""Reproducible B1 HTTP inference comparison: MLX-LM, MLX-One and oMLX.

No weights are downloaded. One server owns the GPU at a time; every round
starts a fresh process, warms it up, and records raw SSE, usage and wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("mlx-lm", "mlx-one", "omlx")
MODELS = (
    (
        "qwen2",
        "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
        "6474d4c1a05bd7bd634a9c993560e4e7272eda3d",
        "8-bit",
        "float16",
    ),
    (
        "qwen3",
        "mlx-community/Qwen3-0.6B-8bit",
        "11de96878523501bcaa86104e3c186de07ff9068",
        "8-bit",
        "bfloat16",
    ),
    (
        "smollm2",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "31b70e2e869a7173562077fd711b654946d38674",
        "BF16",
        "bfloat16",
    ),
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def make_prompt(tokenizer, width, index):
    """Find an exact raw-text/token round trip without adding special tokens."""
    seed = (
        f"Experiment {index:04d}. The research team explored the old city and kept "
        "a detailed journal of its discoveries. Every morning they examined the "
        "streets, compared observations, and described the people they met. "
    )
    ids = tokenizer.encode(seed * (width + 1), add_special_tokens=False).ids
    for offset in range(32):
        selected = ids[offset : offset + width]
        text = tokenizer.decode(selected, skip_special_tokens=False)
        if tokenizer.encode(text, add_special_tokens=False).ids == selected:
            # These three checkpoints have no automatic BOS postprocessor.
            if tokenizer.encode(text, add_special_tokens=True).ids != selected:
                continue
            return {
                "text": text,
                "token_ids": selected,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
    raise ValueError("cannot construct an identical raw-text prompt")


def prepare(args):
    from tokenizers import Tokenizer

    directory = args.directory.resolve()
    models = []
    for key, repo, revision, precision, dtype in MODELS:
        snapshot = args.hub_cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        if not snapshot.is_dir():
            raise FileNotFoundError(f"download the pinned checkpoint first: {repo}@{revision}")
        weights = sorted(snapshot.glob("*.safetensors"))
        if not weights or not all(path.is_file() for path in weights):
            raise FileNotFoundError(f"incomplete safetensors checkpoint: {snapshot}")
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if any(Path(name).name != name or not (snapshot / name).is_file() for name in shards):
                raise ValueError("missing or unsafe checkpoint shards")
        tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        assets = weights + [
            snapshot / "config.json",
            snapshot / "tokenizer.json",
            snapshot / "tokenizer_config.json",
        ]
        if index.exists():
            assets.append(index)
        model = {
            "key": key,
            "repo": repo,
            "revision": revision,
            "precision": precision,
            "execution_dtype": dtype,
            "path": str(snapshot.resolve()),
            "files_sha256": {p.name: digest(p) for p in assets},
            "prompts": [make_prompt(tokenizer, args.prompt_tokens, i) for i in range(args.cases)],
        }
        models.append(model)
        print(
            f"Prepared {repo}: {len(weights)} weight file(s), {args.prompt_tokens}-token prompts",
            flush=True,
        )
    manifest = {
        "schema_version": "competitors-models-v1",
        "prompt_tokens": args.prompt_tokens,
        "models": models,
    }
    save(directory / "models.json", manifest)
    return 0


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def server_command(backend, python, model, folder, port):
    common = ["--host", "127.0.0.1", "--port", str(port)]
    if backend == "mlx-lm":
        return [
            str(python),
            "-m",
            "mlx_lm.server",
            "--model",
            model["path"],
            *common,
            "--prompt-cache-size",
            "0",
            "--prefill-step-size",
            "512",
            "--decode-concurrency",
            "1",
            "--prompt-concurrency",
            "1",
        ]
    if backend == "mlx-one":
        cache_type = "f16" if model["execution_dtype"] == "float16" else "bf16"
        return [
            str(python.parent / "mlx-one"),
            "serve",
            model["path"],
            *common,
            "--offline",
            "--alias",
            model["key"],
            "--inference-mode",
            "reference_dense",
            "--no-cache-prompt",
            "--prefix-cache-mib",
            "0",
            "--no-warmup",
            "--reasoning",
            "off",
            "--parallel",
            "1",
            "--prefill-chunk-size",
            "512",
            "--execution-dtype",
            model["execution_dtype"],
            "--cache-type-k",
            cache_type,
            "--cache-type-v",
            cache_type,
        ]
    models = folder / "models"
    models.mkdir()
    (models / model["key"]).symlink_to(model["path"], target_is_directory=True)
    return [
        str(python.parent / "omlx"),
        "serve",
        "--model-dir",
        str(models),
        *common,
        "--base-path",
        str(folder / "omlx-state"),
        "--no-cache",
        "--no-hf-cache",
        "--headless",
        "--max-concurrent-requests",
        "1",
        "--api-key",
        "local-comparison-key",
    ]


class RSSMonitor:
    """Sample RSS, not MLX allocated memory; model load/warmup are included."""

    def __init__(self, pid):
        self.pid, self.peak = pid, 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.observe, daemon=True)

    def observe(self):
        import psutil

        while not self.stop_event.is_set():
            try:
                process = psutil.Process(self.pid)
                value = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        value += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak = max(self.peak, value)
            except psutil.Error:
                pass
            self.stop_event.wait(0.02)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=2)


@contextmanager
def launch(command, folder, timeout, headers, base_url):
    import httpx

    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMLX_HEADLESS="1",
    )
    save(folder / "command.json", {"command": command, "offline": True})
    log = (folder / "server.log").open("w")
    process = subprocess.Popen(
        command,
        cwd=folder,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.perf_counter()
    try:
        with RSSMonitor(process.pid) as monitor:
            with httpx.Client(timeout=2, trust_env=False, headers=headers) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("server exited; see " + str(folder / "server.log"))
                    try:
                        response = client.get(base_url + "/v1/models")
                        if response.status_code == 200:
                            model_list = response.json()["data"]
                            break
                    except (httpx.HTTPError, ValueError, KeyError):
                        pass
                    if time.perf_counter() - started > timeout:
                        raise TimeoutError("server did not become ready")
                    time.sleep(0.15)
            yield process, monitor, model_list, time.perf_counter() - started
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log.close()


def read_stream(client, url, body, expected_prompt_tokens):
    """Measure first visible text and complete response; never treat SSE count as tokens."""
    started = time.perf_counter()
    first_text = last_text = None
    text, chunks, usage, metrics = "", [], None, None
    finish, done = None, False
    with client.stream("POST", url, json=body) as response:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.read().decode()[:1200]}")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            now = time.perf_counter()
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            chunks.append({"elapsed_seconds": now - started, "data": chunk})
            if "error" in chunk:
                raise RuntimeError("SSE error: " + json.dumps(chunk["error"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("mlx"):
                metrics = chunk["mlx"]
            for choice in chunk.get("choices", []):
                piece = choice.get("text", "") or choice.get("delta", {}).get("content", "")
                if piece:
                    if first_text is None:
                        first_text = now
                    last_text = now
                    text += piece
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    elapsed = time.perf_counter() - started
    if not done or first_text is None or finish is None:
        raise ValueError("incomplete stream, empty text, or missing finish reason")
    if usage is None and metrics:
        usage = {
            "prompt_tokens": metrics.get("prompt_tokens"),
            "completion_tokens": metrics.get("generated_tokens"),
        }
    if not usage or usage.get("prompt_tokens") != expected_prompt_tokens:
        raise ValueError(f"prompt token mismatch: expected {expected_prompt_tokens}, usage={usage}")
    count = usage.get("completion_tokens")
    if type(count) is not int or count <= 0:
        raise ValueError("missing authoritative completion-token count")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if metrics:
        cached = max(
            cached or 0,
            metrics.get("cached_tokens") or 0,
            metrics.get("reused_prefix_tokens") or 0,
            metrics.get("reused_prompt_tokens") or 0,
        )
    if cached:
        raise ValueError("cross-request prompt caching contaminated a cold-prefill sample")
    return {
        "ttft_ms": (first_text - started) * 1000,
        "latency_seconds": elapsed,
        "output_tokens_per_second": count / elapsed,
        "effective_prefill_tokens_per_second": expected_prompt_tokens / (first_text - started),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": count,
        "finish_reason": finish,
        "cached_tokens": cached or 0,
        "text": text,
        "usage": usage,
        "server_metrics": metrics,
        "events": chunks,
        "last_text_seconds": last_text - started,
    }


def environment_metadata(directory):
    metadata = json.loads((directory / "environments.json").read_text())
    for key in BACKENDS:
        if key not in metadata["environments"]:
            raise ValueError("finish environment setup first")
    return metadata


def run(args):
    import httpx

    directory = args.directory.resolve()
    manifest = json.loads((directory / "models.json").read_text())
    environments = environment_metadata(directory)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src/mlx_one").rglob("*.py"))
    }
    hardware = {}
    for name in ["machdep.cpu.brand_string", "hw.memsize", "hw.model"]:
        result = subprocess.run(["sysctl", "-n", name], text=True, capture_output=True)
        hardware[name] = result.stdout.strip() if result.returncode == 0 else None
    receipt = {
        "schema_version": "competitor-http-comparison-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware": hardware,
        "environments": environments,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": source,
        "runner_sha256": digest(__file__),
        "protocol": {
            "rounds": args.rounds,
            "repetitions_per_round": args.repetitions,
            "warmups_per_process": args.warmups,
            "output_budget": args.max_tokens,
            "prompt_tokens": manifest["prompt_tokens"],
            "concurrency": 1,
            "temperature": 0,
            "top_p": 1,
            "cross_request_cache": False,
            "endpoint": "/v1/completions",
            "timing": "HTTP SSE wall time",
        },
        "models": manifest["models"],
        "sessions": [],
    }
    save(output / "results.json", receipt)
    selected = [m for m in manifest["models"] if m["key"] in args.models]
    for model_index, model in enumerate(selected):
        # Hash all actual weights again before GPU measurements.
        for name, expected in model["files_sha256"].items():
            if digest(Path(model["path"]) / name) != expected:
                raise ValueError("changed checkpoint asset: " + name)
        for round_index in range(args.rounds):
            offset = (model_index + round_index) % len(BACKENDS)
            order = BACKENDS[offset:] + BACKENDS[:offset]
            for backend in order:
                if backend not in args.backends:
                    continue
                folder = output / f"{model['key']}-{round_index + 1}-{backend}"
                folder.mkdir()
                port = free_port()
                base_url = f"http://127.0.0.1:{port}"
                python = Path(environments["environments"][backend]["python"])
                command = server_command(backend, python, model, folder, port)
                session = {
                    "model": model["key"],
                    "backend": backend,
                    "round": round_index + 1,
                    "folder": str(folder),
                    "command": command,
                    "samples": [],
                    "warmups": [],
                    "status": "RUNNING",
                }
                receipt["sessions"].append(session)
                print(f"{model['key']} round {round_index + 1}: {backend}", flush=True)
                headers = {"Authorization": "Bearer local-comparison-key"}
                try:
                    with launch(command, folder, args.startup_timeout, headers, base_url) as state:
                        process, monitor, model_list, readiness = state
                        if not model_list:
                            raise ValueError("server model inventory is empty")
                        model_id = model["path"] if backend == "mlx-lm" else model["key"]
                        session.update(model_id=model_id, ready_seconds=readiness)
                        with httpx.Client(
                            timeout=args.request_timeout, trust_env=False, headers=headers
                        ) as client:
                            for i in range(args.warmups + args.repetitions):
                                index = round_index * (args.warmups + args.repetitions) + i
                                prompt = model["prompts"][index]
                                body = {
                                    "model": model_id,
                                    "prompt": prompt["text"],
                                    "temperature": 0,
                                    "top_p": 1,
                                    "seed": 0,
                                    "max_tokens": args.max_tokens,
                                    "stream": True,
                                }
                                if backend != "mlx-one":
                                    body["stream_options"] = {"include_usage": True}
                                sample = read_stream(
                                    client,
                                    base_url + "/v1/completions",
                                    body,
                                    len(prompt["token_ids"]),
                                )
                                sample.update(
                                    prompt_index=index,
                                    prompt_sha256=prompt["sha256"],
                                    request_body=body,
                                )
                                if sample["completion_tokens"] > args.max_tokens:
                                    raise ValueError("server exceeded the requested output budget")
                                kind = "warmups" if i < args.warmups else "samples"
                                session[kind].append(sample)
                                save(folder / f"{kind}-{i + 1}.json", sample)
                                print(
                                    f"  {kind}: {sample['completion_tokens']} tokens, "
                                    f"TTFT {sample['ttft_ms']:.1f} ms, "
                                    f"{sample['output_tokens_per_second']:.1f} tok/s",
                                    flush=True,
                                )
                        session["peak_rss_bytes"] = monitor.peak
                    session["status"] = "PASS"
                except Exception as error:
                    session.update(status="FAIL", error=f"{type(error).__name__}: {error}")
                    print("  FAIL: " + session["error"], flush=True)
                save(output / "results.json", receipt)
                render_report(receipt, args.report.resolve(), output)
                time.sleep(args.cooldown)
    render_report(receipt, args.report.resolve(), output)
    return 0 if all(s["status"] == "PASS" for s in receipt["sessions"]) else 1


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def comparison_analysis(receipt, grouped):
    """Retain EOS samples, but compare fixed-length speed only on shared cases."""
    rows = [
        "## Matched full-length comparisons",
        "",
        "This view keeps a prompt only when **all three runtimes** generated the requested "
        "token budget and finished by length. Excluded EOS cases remain in the complete results "
        "above. No runtime is allowed to choose its own subset.",
        "",
        "| Model | Runtime | Shared full-length samples | TTFT ms | Latency s | End-to-end tok/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    parity_rows, findings, analysis = [], [], {}
    budget = receipt["protocol"]["output_budget"]
    expected = receipt["protocol"]["rounds"] * receipt["protocol"]["repetitions_per_round"]
    for model in receipt["models"]:
        by_backend = {
            backend: {sample["prompt_index"]: sample for sample in grouped[(model["key"], backend)]}
            for backend in BACKENDS
        }
        common = set.intersection(*(set(values) for values in by_backend.values()))
        full = sorted(
            index
            for index in common
            if all(
                values[index]["completion_tokens"] == budget
                and values[index]["finish_reason"] == "length"
                for values in by_backend.values()
            )
        )
        analysis[model["key"]] = {
            "common_prompt_indices": sorted(common),
            "full_length_prompt_indices": full,
        }
        rates = {}
        for backend, values in by_backend.items():
            if full:
                selected = [values[index] for index in full]
                ttft = statistics.median(sample["ttft_ms"] for sample in selected)
                latency = statistics.median(sample["latency_seconds"] for sample in selected)
                rates[backend] = statistics.median(
                    sample["output_tokens_per_second"] for sample in selected
                )
                rows.append(
                    f"| {model['repo']} | {backend} | {len(full)} | {ttft:.2f} | "
                    f"{latency:.3f} | {rates[backend]:.2f} |"
                )
            else:
                rows.append(f"| {model['repo']} | {backend} | 0 | — | — | — |")
            if backend == "mlx-lm":
                continue
            exact = sum(values[i]["text"] == by_backend["mlx-lm"][i]["text"] for i in common)
            trimmed = sum(
                values[i]["text"].strip() == by_backend["mlx-lm"][i]["text"].strip() for i in common
            )
            parity_rows.append(
                f"| {model['repo']} | {backend} | {exact}/{len(common)} | {trimmed}/{len(common)} |"
            )
        if rates and len(common) == expected:
            fastest = max(rates, key=rates.get)
            findings.append(
                f"- **{model['repo']}**: {fastest} had the highest median "
                f"full-length end-to-end rate ({rates[fastest]:.2f} tok/s) "
                f"on {len(full)} shared prompts in this run."
            )
        excluded = sorted(common - set(full))
        if excluded:
            findings.append(
                f"- {model['repo']}: prompt indices {excluded} are excluded from "
                "the fixed-length view because output lengths/stopping differ. "
                "See raw usage, finish reasons and text for all runtimes."
            )
    rows += [
        "",
        "### Findings",
        "",
        *findings,
        "",
        "Small sample sizes and differing dependency versions limit generalization. "
        "These rankings apply to this B1 HTTP workload, not all model sizes or concurrency levels.",
        "",
        "## Output agreement",
        "",
        "Visible text compared against MLX-LM for the same prompt; whitespace trimming "
        "removes only leading/trailing whitespace. This is a diagnostic, not an accuracy score.",
        "",
        "| Model | Runtime | Exact matches | Matches after trimming edge whitespace |",
        "| --- | --- | ---: | ---: |",
        *parity_rows,
        "",
        "The complete raw texts are retained. A matching token budget does not establish "
        "matching predictions. Backend usage counters can also differ around EOS: "
        "compare actual text and finish reasons before interpreting short-output token rates.",
    ]
    return rows, analysis


def render_report(receipt, destination, output):
    grouped = defaultdict(list)
    sessions = defaultdict(list)
    for session in receipt["sessions"]:
        key = session["model"], session["backend"]
        sessions[key].append(session)
        if session["status"] == "PASS":
            grouped[key].extend(session["samples"])
    p = receipt["protocol"]
    rows = [
        "# MLX-LM vs MLX-One vs oMLX: text generation performance",
        "",
        f"Measured {receipt['created_at_utc']} on "
        f"{receipt['hardware'].get('machdep.cpu.brand_string')}, "
        f"{int(receipt['hardware'].get('hw.memsize') or 0) / 1024**3:.0f} GiB RAM; "
        f"{receipt['platform']}; Python {receipt['python']}.",
        "",
        "## Scope and protocol",
        "",
        "This is a single-request, end-to-end **HTTP server** comparison. It does not "
        "measure bare model kernels, concurrent throughput, chat templating, or cache reuse.",
        "",
        f"- Identical immutable local checkpoints and {p['prompt_tokens']}-token raw prompts; "
        "full weight/tokenizer hashes and exact prompt IDs are recorded.",
        f"- Greedy decoding (temperature 0, top_p 1), maximum {p['output_budget']} generated "
        "tokens, natural EOS respected, no chat template or speculative decoding.",
        f"- {p['rounds']} independent server processes per model/backend, each with "
        f"{p['warmups_per_process']} excluded warmup(s) and "
        f"{p['repetitions_per_round']} timed request(s). Runtime order rotates between rounds "
        "and models; servers never overlap.",
        "- Prefix/prompt caches disabled; responses with reported cache hits or incorrect "
        "prompt-token counts are rejected. Model loading and warmup are excluded from timing.",
        "- MLX-One uses native `reference_dense`, checkpoint floating dtype and matching "
        "16-bit KV dtype. oMLX uses its stock single-request batching path with SSD cache off. "
        "MLX-LM has prompt cache off and concurrency 1.",
        "- **TTFT** is request send to first nonempty text fragment, including HTTP, "
        "scheduling, tokenization, prefill and text buffering. **End-to-end tok/s** is "
        "the server's actual completion-token count divided by full response wall time; "
        "it includes prefill and is not decode-only throughput. "
        "SSE chunks are never counted as tokens.",
        "- **Peak RSS** samples the server and children every 20 ms, including model load, "
        "warmup and timed requests. It is process RSS, not MLX peak allocation or total "
        "physical unified-memory use. Sampled peaks may miss brief allocations.",
        "",
        "## Versions",
        "",
        "| Runtime | Package | MLX / Metal | MLX-LM dependency |",
        "| --- | --- | --- | --- |",
    ]
    for backend in BACKENDS:
        packages = receipt["environments"]["environments"][backend]["packages"]
        rows.append(
            f"| {backend} | {packages.get(backend, 'unknown')} | "
            f"{packages.get('mlx', '?')} / {packages.get('mlx-metal', '?')} | "
            f"{packages.get('mlx-lm', 'not installed')} |"
        )
    rows += [
        "",
        "The supported package environments resolve different MLX versions. "
        "oMLX also pins its own MLX-LM revision. These are **whole-stack results**, "
        "so differences cannot be attributed solely to the server or model implementation. "
        "Each isolated environment passed `pip check`; full package inventories are retained.",
        "",
        "## Results",
        "",
        "Per-column medians; latency p95 uses linear interpolation over the measured samples. "
        "Peak RSS is the maximum across independent processes. Short EOS outputs are shown "
        "with their actual lengths and should not be interpreted as fixed-length decode speed.",
        "PASS means the measurement protocol completed; it does not mean output equivalence. "
        "The p95 values are descriptive estimates from a small sample, not a tail-latency SLA.",
        "",
        "| Model | Precision | Runtime | Samples | Output tokens (range) | TTFT ms | "
        "Latency s | Latency p95 s | End-to-end tok/s | Peak RSS GiB | Status |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in receipt["models"]:
        for backend in BACKENDS:
            key = model["key"], backend
            entries = sessions[key]
            values = grouped[key]
            complete = (
                len(entries) == p["rounds"]
                and all(s["status"] == "PASS" for s in entries)
                and len(values) == p["rounds"] * p["repetitions_per_round"]
            )
            if not values:
                status = "FAIL" if any(s["status"] == "FAIL" for s in entries) else "NOT RUN"
                rows.append(
                    f"| {model['repo']} | {model['precision']} | {backend} | 0 | "
                    f"— | — | — | — | — | — | {status} |"
                )
                continue
            count = [v["completion_tokens"] for v in values]
            medians = {
                key: statistics.median(value[key] for value in values)
                for key in ["ttft_ms", "latency_seconds", "output_tokens_per_second"]
            }
            peak = max(s.get("peak_rss_bytes", 0) for s in entries) / 1024**3
            rows.append(
                f"| {model['repo']} | {model['precision']} | {backend} | {len(values)} | "
                f"{min(count)}–{max(count)} | {medians['ttft_ms']:.2f} | "
                f"{medians['latency_seconds']:.3f} | "
                f"{percentile([v['latency_seconds'] for v in values], 0.95):.3f} | "
                f"{medians['output_tokens_per_second']:.2f} | {peak:.3f} | "
                f"{'PASS' if complete else 'PARTIAL'} |"
            )
    additional_rows, _ = comparison_analysis(receipt, grouped)
    rows += ["", *additional_rows, ""]
    rows += [
        "",
        "## Model identity",
        "",
        "| Model | Revision | Floating / KV dtype |",
        "| --- | --- | --- |",
    ]
    for model in receipt["models"]:
        rows.append(f"| {model['repo']} | `{model['revision']}` | {model['execution_dtype']} |")
    failed = [s for s in receipt["sessions"] if s["status"] == "FAIL"]
    if failed:
        rows += ["", "## Failed attempts", ""]
        for session in failed:
            rows.append(
                f"- {session['model']} / {session['backend']} / round {session['round']}: "
                f"{session['error']}"
            )
    try:
        evidence = output.relative_to(ROOT)
    except ValueError:
        evidence = output
    rows += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd /path/to/mlx-one",
        "python3.12 scripts/compare_competators_setup.py",
        "client_python=local/competitor-comparison/venvs/client/bin/python",
        '"$client_python" scripts/compare_competators.py prepare',
        '"$client_python" scripts/compare_competators.py run '
        "--output local/competitor-comparison/runs/NEW-RUN --report compare_competators.md",
        "```",
        "",
        "Run native servers on an Apple Silicon session with Metal access. Setup installs "
        "dependencies; preparation/runs use already cached, pinned models and never "
        "download weights. All servers bind only to loopback and use separate state directories.",
        "",
        "## Evidence and limits",
        "",
        "- Raw results, SSE payloads, generated text, commands, server logs "
        f"and source hashes: `{evidence}`.",
        f"- Machine-readable manifest: `{evidence / 'results.json'}`.",
        f"- Current MLX-One source commit: `{receipt['source_commit']}`; "
        "exact working-tree "
        "production hashes are in the manifest. "
        "This benchmark does not promote model qualification.",
        "- No corpus-quality grading, power measurement, background-load isolation or "
        "thermal instrumentation was performed. "
        "Repeated output does not establish numerical parity.",
        "- Official implementations: [MLX-LM](https://github.com/ml-explore/mlx-lm), "
        "[MLX-One](https://github.com/SSusantAchary/mlx-one), "
        "[oMLX](https://github.com/jundot/omlx).",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "local/competitor-comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--hub-cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    prep.add_argument("--prompt-tokens", type=int, default=128)
    prep.add_argument("--cases", type=int, default=24)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--report", type=Path, default=ROOT / "compare_competators.md")
    execute.add_argument("--rounds", type=int, default=3)
    execute.add_argument("--repetitions", type=int, default=2)
    execute.add_argument("--warmups", type=int, default=1)
    execute.add_argument("--max-tokens", type=int, default=128)
    execute.add_argument("--cooldown", type=float, default=2)
    execute.add_argument("--startup-timeout", type=float, default=180)
    execute.add_argument("--request-timeout", type=float, default=180)
    execute.add_argument(
        "--models", nargs="+", choices=[m[0] for m in MODELS], default=[m[0] for m in MODELS]
    )
    execute.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    report = commands.add_parser("report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--report", type=Path, default=ROOT / "compare_competators.md")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.prompt_tokens < 1 or args.cases < 1:
            parser.error("prompt tokens and cases must be positive")
        return prepare(args)
    if args.command == "report":
        render_report(
            json.loads(args.results.read_text()), args.report.resolve(), args.results.parent
        )
        return 0
    if min(args.rounds, args.repetitions, args.warmups, args.max_tokens) < 1:
        parser.error("rounds, repetitions, warmups and token budget must be positive")
    manifest = json.loads((args.directory / "models.json").read_text())
    if any(
        len(model["prompts"]) < args.rounds * (args.warmups + args.repetitions)
        for model in manifest["models"]
    ):
        parser.error("prepare more prompt cases before running")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
