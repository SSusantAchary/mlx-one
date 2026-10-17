"""Isolated native reference Engine cases; invoked only by the supervised harness."""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import signal
import statistics
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from mlx_one.engine.observation import observe_request
from mlx_one.engine.timing import RequestTiming
from mlx_one.p0_engine_qualification import SCHEMA, atomic_json, exact_prompt, utc

CORPUS = (
    "The future of local computing is",
    "What is the capital of France?",
    "Write a Python function that adds two integers.",
    'Return JSON with keys "name" and "count".',
    "What is 17 plus 25? Explain briefly.",
    "User: My name is Mira. Assistant: Hello Mira. User: What is my name? Assistant:",
    "Explain café, naïve, and résumé.",
    "नमस्ते। एक छोटा वाक्य लिखिए।",
    "Describe these symbols: 🌍 🚀 🧠.",
    "one one one two two two three",
    " Spaces, commas; punctuation!\n\nContinue: ",
    "Answer briefly: why is the sky blue?",
)


@contextmanager
def deadline(seconds):
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(_number, _frame):
        raise TimeoutError(f"case exceeded {seconds}s deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL,
                     min(seconds, previous_timer[0]) if previous_timer[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL,
                             max(previous_timer[0] - (time.monotonic() - started), .001))


def component_state(cache, *, arrays=False):
    import mlx.core as mx
    import numpy as np

    result = []
    for entry in cache:
        fields = {}
        for name in ("keys", "values", "_keys", "_values", "conv_state", "recurrent_state"):
            value = getattr(entry, name, None)
            if hasattr(value, "dtype"):
                # Only committed KV, never uninitialized rounded allocation capacity.
                if value.ndim == 4 and name in {"keys", "values", "_keys", "_values"}:
                    value = value[:, :, :entry.offset]
                mx.eval(value)
                fields[name] = {"shape": list(value.shape),
                                "dtype": str(value.dtype).rsplit(".", 1)[-1]}
                if arrays:
                    fields[name]["array"] = np.asarray(value.astype(mx.float32)).copy()
        result.append({"class": type(entry).__name__, "position": entry.offset,
                       "bytes": int(getattr(entry, "nbytes", 0)), "fields": fields})
    return result


class Observer:
    def __init__(self, *, capture=False, admission=False):
        self.timing = RequestTiming(time.perf_counter(), trace=True)
        self.capture_admission = admission
        self.admissions = []
        self.capture_state = capture
        self.capture_bytes = True
        self.peak_cache_bytes = 0
        self.states = []
        self.batch_sizes = []

    def __call__(self, event):
        if event["kind"] == "admission":
            record = event["diagnostics"]
            if (len(self.admissions) < 16 or record["decision"] != "admit"
                    or record.get("attempt", 0)):
                self.admissions.append(record)
        elif event["kind"] == "cache_bytes":
            self.peak_cache_bytes = max(self.peak_cache_bytes, event["bytes"])
        elif event["kind"] == "state":
            import mlx.core as mx
            import numpy as np

            self.states.append({"input_ids": list(event["input_ids"]),
                "logits": np.asarray(event["logits"][0, -1].astype(mx.float32)).copy(),
                "components": component_state(event["cache"], arrays=True)})
        else:
            self.timing.observe(event)
            if event["kind"] == "forward":
                self.batch_sizes.append(event["batch_size"])

    def summary(self):
        return {**self.timing.summary(), "token_ids": list(self.timing.token_ids),
                "forward_ms": self.timing.forward_ms,
                "actual_prefill_tokens": self.timing.prefill_tokens,
                "tokenization_ms": self.timing.tokenization_ms,
                "batch_sizes": self.batch_sizes, "peak_request_state_bytes": self.peak_cache_bytes,
                "admission_diagnostics": self.admissions}


def error_metrics(expected, actual, dtype="float16"):
    import numpy as np

    from mlx_one.inference_baseline import TOLERANCES

    expected, actual = np.asarray(expected), np.asarray(actual)
    difference = np.abs(expected - actual)
    atol, rtol = TOLERANCES[dtype]
    finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
    scale = np.maximum(np.abs(expected), atol)
    return {"max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "rms_error": float(np.sqrt(np.mean(difference**2))),
            "max_scaled_relative_error": float((difference / scale).max()),
            "greedy_agreement": bool(np.argmax(expected) == np.argmax(actual)),
            "within_tolerance": finite and bool(
                np.allclose(expected, actual, atol=atol, rtol=rtol)),
            "atol": atol, "rtol": rtol, "precision": dtype}


def owned_clone(executor, key, cache):
    """Lazy cache tensors must be constructed and consumed by the same device owner."""
    return executor.call(key, cache.clone)


def native_parameters(model):
    from mlx.utils import tree_flatten

    quantized, affine_metadata = {}, set()
    for _, module in model.named_modules():
        if hasattr(module, "bits"):
            quantized[id(module.weight)] = module.bits
            affine_metadata.update((id(module.scales), id(module.biases)))
    count, physical, seen = 0, 0, set()
    for _, value in tree_flatten(model.parameters()):
        if id(value) in seen:
            continue
        seen.add(id(value))
        physical += value.nbytes
        if id(value) not in affine_metadata:
            count += (value.size * 32 // quantized[id(value)]
                      if id(value) in quantized else value.size)
    return {"logical_parameters": count, "physical_model_bytes": physical,
            "quantized_modules": len(quantized), "unique_parameter_tensors": len(seen)}


class Cases:
    def __init__(self, manifest, directory):
        import mlx.core as mx

        from mlx_one import Engine
        from mlx_one.engine.cache_specs import CacheConfig

        self.manifest, self.directory = manifest, directory
        self.metadata = manifest["metadata"]
        self.rows = {"schema_version": SCHEMA, "cases": [], "benchmarks": [], "memory": [],
                     "parity": [], "admission": []}
        self.mx = mx
        mx.set_default_device(mx.gpu)
        assert mx.metal.is_available()
        mx.reset_peak_memory()
        self.load_started = time.perf_counter()
        ceiling = manifest["environment"]["preflight"]["process_ceiling_bytes"]
        budget = min(manifest["environment"]["policy"]["cache_memory_budget_bytes"],
                     ceiling - max(self.metadata["weight_file_bytes"], 1024**3) - 2 * 1024**3)
        if budget <= 0:
            raise MemoryError("no safe cache/transient budget below the process ceiling")
        self.engine = Engine(self.metadata["snapshot_path"],
            revision=self.metadata["revision"], offline=True, inference_mode="reference_dense",
            execution_dtype="float16", cache_config=CacheConfig(backend="dense",
                prefix_cache_enabled=False,
                memory_budget_bytes=budget))
        mx.synchronize()
        self.load_seconds = time.perf_counter() - self.load_started
        self.load_peak_bytes = mx.get_peak_memory()
        self.bundle = self.engine.model
        self.initial_model_bytes = mx.get_active_memory()
        self.save()

    def save(self):
        atomic_json(self.directory / "results.json", self.rows)

    def check(self, case, function, *, seconds=120, applicable=True):
        row = {"case_id": case, "timestamp": utc(), "model_id": self.metadata["model_id"],
               "revision": self.metadata["revision"], "applicable": applicable}
        started = time.perf_counter()
        previous_case = getattr(self, "_current_case", None)
        self._current_case = case
        try:
            with deadline(seconds):
                value = function()
            row.update(status="PASS", details=value)
        except NotImplementedError as exc:
            row.update(status="NOT_SUPPORTED", reason=str(exc))
        except Exception as exc:
            row.update(status="FAIL", error_type=type(exc).__name__, reason=str(exc),
                       traceback=traceback.format_exc(),
                       shared_blocker=case.startswith(("F", "G")))
        self._current_case = previous_case
        row["duration_seconds"] = time.perf_counter() - started
        self.rows["cases"].append(row)
        self.save()
        print(f"{case} {row['status']}", flush=True)
        return row

    def prompt(self, text, **options):
        if self.bundle.architecture == "gpt2":
            return text
        return self.bundle.chat_template.render([{"role": "user", "content": text}], **options)

    def generate(self, prompt, *, capture=False, stream=False, admission=False, **options):
        observer = Observer(capture=capture, admission=admission)
        seconds = 600 if len(self.bundle.tokenizer.encode(prompt)) >= 2048 else 120
        engine_started = time.perf_counter()
        observer.timing.started = engine_started
        with observe_request(observer), deadline(seconds):
            try:
                if stream:
                    chunks = list(self.engine.stream(prompt, **options))
                    result = {"text": "".join(c.text for c in chunks),
                              "finish_reason": chunks[-1].finish_reason,
                              "generation_tokens": sum(c.token_id is not None for c in chunks),
                              "chunk_ids": [c.token_id for c in chunks if c.token_id is not None]}
                else:
                    value = self.engine.generate(prompt, **options)
                    result = value.to_dict()
            finally:
                self.rows["admission"].extend({
                    **record, "loaded_model_id": record["model_id"],
                    "model_id": self.metadata["model_id"],
                    "source": self.manifest["environment"]["source"],
                    "case_id": getattr(self, "_current_case", None)}
                    for record in observer.admissions)
        result["engine_wall_seconds"] = time.perf_counter() - engine_started
        result["observer"] = observer.summary()
        assert result["generation_tokens"] == len(result["observer"]["token_ids"])
        expected = self.bundle.tokenizer.decode(result["observer"]["token_ids"])
        if result["text"] != expected:
            atomic_json(self.directory / f"text-mismatch-{time.time_ns()}.json",
                        {"result": result, "expected": expected, "prompt": prompt})
            raise AssertionError(
                "incremental text does not equal decoding the complete accepted token sequence")
        assert not observer.batch_sizes or set(observer.batch_sizes) == {1}
        assert self.engine._cache_manager.stats().active_handles == 0
        return result, observer

    def startup(self):
        def forward():
            result, observer = self.generate(self.prompt(CORPUS[0]), capture=True, max_tokens=16)
            assert observer.states and result["generation_tokens"] > 0
            self.first_result, self.first_observer = result, observer
            return result

        checks = {
            "A01": lambda: self.metadata["snapshot_path"],
            "A02": lambda: {"validated_assets": self.metadata["assets"]},
            "A03": lambda: {"load_seconds": self.load_seconds,
                "resident_active_bytes": self.initial_model_bytes,
                "load_peak_bytes": self.load_peak_bytes,
                "model_class": type(self.bundle.model).__module__ + "." +
                               type(self.bundle.model).__name__,
                "observed_parameters": native_parameters(self.bundle.model),
                "active_text_parameters": native_parameters(
                    getattr(self.bundle.model, "language_model", self.bundle.model))},
            "A04": lambda: self.require(self.bundle.architecture == self.metadata["architecture"]),
            "A05": lambda: self.require(self.bundle.quantization == self.metadata["quantization"]),
            "A06": lambda: self.require(self.engine._inference_mode == "reference_dense" and
                                         self.engine._actor is None),
            "A07": lambda: self.require(bool(self.bundle.tokenizer.encode("hello"))),
            "A08": forward,
            "A09": lambda: self.require(self.first_result["generation_tokens"] == 16),
            "A10": lambda: self.finite(self.first_observer),
            "A11": self.engine.capabilities,
            "A12": self.engine.runtime_stats,
            "A13": lambda: self.require(self.bundle.revision == self.metadata["revision"] and
                self.bundle.path.resolve() == Path(self.metadata["snapshot_path"]).resolve(),
                {"checkpoint_id": self.metadata["model_id"], "loaded_id": self.bundle.model_id,
                 "revision": self.bundle.revision, "snapshot_path": str(self.bundle.path)}),
            "A14": self.close,
        }
        for case, function in checks.items():
            self.check(case, function)

    @staticmethod
    def require(condition, detail=None):
        if not condition:
            raise AssertionError(detail or "qualification assertion failed")
        return detail or {"verified": True}

    def finite(self, observer):
        import numpy as np

        assert observer.states
        assert all(np.isfinite(state["logits"]).all() for state in observer.states)
        return {"evaluated_forward_count": len(observer.states)}

    def correctness(self):
        def deterministic(sampled=False):
            data = []
            for index, text in enumerate(CORPUS):
                prompt = self.prompt(text, enable_thinking=False)
                settings = {"max_tokens": 128, "temperature": .7 if sampled else 0,
                            "top_p": .9 if sampled else 1, "seed": 123,
                            "top_k": 40 if sampled else None}
                runs = [self.generate(prompt, **settings)[0] for _ in range(5)]
                assert all(r["observer"]["token_ids"] == runs[0]["observer"]["token_ids"]
                           and r["text"] == runs[0]["text"] for r in runs)
                data.append({"corpus_index": index, "prompt": prompt, "settings": settings,
                             "runs": runs})
            path = self.directory / ("sampling.json" if sampled else "greedy.json")
            atomic_json(path, data)
            return {"corpus_cases": len(data), "repetitions": 5, "artifact": str(path)}

        self.check("B01", deterministic, seconds=7200)
        self.check("B02", lambda: deterministic(True), seconds=7200)

        def output_limits():
            runs = [self.generate(self.prompt(CORPUS[0]), max_tokens=n)[0]
                    for n in (1, 16, 64, 128)]
            assert all(r["generation_tokens"] <= n
                       for r, n in zip(runs, (1, 16, 64, 128), strict=True))
            return runs

        self.check("B03", output_limits)

        def termination():
            import mlx_one.text.generation as generation

            prompt = self.prompt(CORPUS[0])
            base, _ = self.generate(prompt, max_tokens=32)
            assert base["generation_tokens"] > 3
            ids = base["observer"]["token_ids"]
            stop = self.bundle.tokenizer.decode(ids[:3])
            assert stop and self.bundle.tokenizer.encode(stop) == ids[:3]
            runs = [self.generate(prompt, max_tokens=32, stop=stops)[0]
                    for stops in ((stop,), (stop, stop + "!"), (stop + "!", stop))]
            assert all(r["finish_reason"] == "stop" and r["text"] == "" for r in runs)
            eos = self.bundle.eos_token_ids or (self.bundle.tokenizer.eos_token_id,)
            assert eos[0] is not None
            previous = generation.select_token
            try:
                generation.select_token = lambda *_args: eos[0]
                controlled, _ = self.generate(prompt, max_tokens=4)
            finally:
                generation.select_token = previous
            assert controlled["finish_reason"] == "stop" and controlled["generation_tokens"] == 0
            return {"stop_crosses_token_events": len(ids[:3]), "stop_cases": runs,
                    "controlled_eos_id": eos[0], "controlled_eos": controlled,
                    "control_scope": "native model evaluated; sampler forced to real EOS ID"}

        self.check("B04", termination)

        def lengths():
            results = []
            for length in (128, 512, 2048, 8192):
                if length + 16 > self.bundle.context_length:
                    results.append({"length": length, "status": "NOT_SUPPORTED",
                                    "reason": "checkpoint context limit"})
                    continue
                prompt = exact_prompt(self.bundle.tokenizer, length)
                assert len(self.bundle.tokenizer.encode(prompt)) == length
                result, _ = self.generate(prompt, max_tokens=16)
                results.append({"length": length, "result": result})
            overflow = exact_prompt(self.bundle.tokenizer, self.bundle.context_length)
            try:
                self.generate(overflow, max_tokens=1)
            except ValueError:
                pass
            else:
                raise AssertionError("context overflow was accepted")
            return results

        self.check("B05", lengths, seconds=1800)

        def templates():
            if self.bundle.architecture == "gpt2":
                raise NotImplementedError("GPT-2 completion does not require chat formatting")
            messages = [{"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                        {"role": "user", "content": "Explain gravity."}]
            rendered = []
            for thinking in (False, True):
                prompt = self.bundle.chat_template.render(messages, enable_thinking=thinking)
                assert all(m["content"] in prompt for m in messages)
                tokens = self.bundle.tokenizer.encode(prompt)
                assert tokens
                result, _ = self.generate(prompt, max_tokens=16)
                rendered.append({"thinking_requested": thinking, "prompt": prompt,
                                 "token_ids": tokens, "result": result})
            return rendered

        self.check("B06", templates, applicable=self.bundle.architecture != "gpt2")

    def streaming(self):
        pairs = []

        def run():
            lengths = [32] * 10 + [128] * 5 + [min(2048, self.bundle.context_length - 128)] * 3
            for index, length in enumerate(lengths):
                prompt = self.prompt(CORPUS[index % len(CORPUS)] + "\n" +
                                     exact_prompt(self.bundle.tokenizer, length),
                                     enable_thinking=False)
                ordinary, _ = self.generate(prompt, max_tokens=64)
                streamed, _ = self.generate(prompt, stream=True, max_tokens=64)
                assert ordinary["observer"]["token_ids"] == streamed["chunk_ids"]
                assert ordinary["text"] == streamed["text"]
                assert ordinary["finish_reason"] == streamed["finish_reason"]
                assert len(streamed["chunk_ids"]) == streamed["generation_tokens"]
                streamed["text"].encode("utf-8", errors="strict")
                pairs.append({"index": index, "ordinary": ordinary, "streamed": streamed})
            path = self.directory / "streaming-pairs.json"
            atomic_json(path, pairs)
            return {"artifact": str(path), "pairs": len(pairs)}

        row = self.check("C01", run, seconds=1800)
        for case in (*range(2, 8), 9):
            self.check(f"C{case:02}", lambda: self.require(
                row["status"] == "PASS", {"verified_pair_artifact": row.get("details")}))

        def eos():
            import mlx_one.text.generation as generation

            eos_ids = self.bundle.eos_token_ids or (self.bundle.tokenizer.eos_token_id,)
            assert eos_ids and eos_ids[0] is not None
            previous = generation.select_token
            try:
                generation.select_token = lambda *_args: eos_ids[0]
                prompt = self.prompt(CORPUS[0])
                ordinary, _ = self.generate(prompt, max_tokens=8)
                streamed, _ = self.generate(prompt, stream=True, max_tokens=8)
            finally:
                generation.select_token = previous
            for value in (ordinary, streamed):
                assert value["finish_reason"] == "stop"
                assert value["text"] == "" and value["generation_tokens"] == 0
                assert value["observer"]["token_ids"] == []
                assert value["observer"]["batch_sizes"] == [1]
            return {"controlled_eos_id": eos_ids[0], "ordinary": ordinary, "streamed": streamed,
                    "control_scope": "native weights evaluated; sampler forced to checkpoint EOS"}

        self.check("C08", eos)

        def early_close():
            stream = self.engine.stream(self.prompt(CORPUS[0]), max_tokens=128)
            next(stream)
            stream.close()
            stats = self.engine._cache_manager.stats()
            assert stats.active_handles == stats.reserved_bytes == 0
            return stats.to_dict()

        self.check("C10", early_close)

    def state(self):
        import numpy as np

        from mlx_one.core.cache import configure_kv_caches
        from mlx_one.engine.cache import CacheBundle

        engine_observer = None
        comparisons, component_comparisons, offsets = [], [], []

        def fresh():
            return CacheBundle(tuple(configure_kv_caches(
                self.bundle.model.make_cache(), "f16", "f16")))

        def forward(cache, ids):
            output = self.bundle.model(self.mx.array([ids]), cache=cache.entries)
            self.mx.eval(output.logits)
            return np.asarray(output.logits[0, -1].astype(self.mx.float32)).copy()

        def compare():
            nonlocal engine_observer
            prompt = self.prompt("A short shared prefix with a divergent continuation.",
                                 enable_thinking=False)
            _, engine_observer = self.generate(prompt, capture=True, max_tokens=4)
            context = list(self.bundle.tokenizer.encode(prompt))
            artifacts = {}
            for index, observed in enumerate(engine_observer.states):
                oracle = fresh()
                try:
                    expected = self.engine._device.call(id(self.engine),
                        lambda c=oracle: forward(c, context))
                    states = self.engine._device.call(id(self.engine),
                        lambda c=oracle: component_state(c.entries, arrays=True))
                    metrics = error_metrics(expected, observed["logits"])
                    comparisons.append(metrics)
                    artifacts[f"oracle_{index}"] = expected
                    artifacts[f"engine_{index}"] = observed["logits"]
                    offsets.append([s["position"] for s in observed["components"]])
                    assert all(s["position"] == len(context) for s in states)
                    for layer, (left, right) in enumerate(zip(
                            states, observed["components"], strict=True)):
                        assert left["position"] == right["position"]
                        for name, field in left["fields"].items():
                            other = right["fields"][name]
                            assert field["dtype"] == other["dtype"]
                            assert field["shape"] == other["shape"]
                            state_metric = error_metrics(field["array"].ravel(),
                                other["array"].ravel(), field["dtype"])
                            component_comparisons.append({"layer": layer, "field": name,
                                                          **state_metric})
                    np.savez_compressed(self.directory / "logits.npz", **artifacts)
                    self.rows["parity"] = [{"model_id": self.metadata["model_id"],
                        "revision": self.metadata["revision"], "logits": comparisons,
                        "components": component_comparisons, "positions": offsets,
                        "prompt_token_ids": self.bundle.tokenizer.encode(prompt),
                        "forced_token_ids": [int(np.argmax(s["logits"]))
                                             for s in engine_observer.states],
                        "artifact": str(self.directory / "logits.npz")}]
                    self.save()
                    assert metrics["within_tolerance"] and metrics["greedy_agreement"], metrics
                    context.append(int(np.argmax(observed["logits"])))
                finally:
                    oracle.release()
            np.savez_compressed(self.directory / "logits.npz", **artifacts)
            self.rows["parity"] = [{"model_id": self.metadata["model_id"],
                "revision": self.metadata["revision"], "logits": comparisons,
                "components": component_comparisons, "positions": offsets,
                "prompt_token_ids": self.bundle.tokenizer.encode(prompt),
                "forced_token_ids": [int(np.argmax(s["logits"])) for s in engine_observer.states],
                "artifact": str(self.directory / "logits.npz")}]
            return self.rows["parity"]

        diagnostic = self.check("D01", compare)
        for index in range(2, 6):
            self.check(f"D{index:02}", lambda: self.require(
                diagnostic["status"] == "PASS" and len(comparisons) == 4,
                {"logit_comparisons": comparisons, "positions": offsets}))

        def masks():
            ids = self.bundle.tokenizer.encode("One two three four five six seven.")
            full, prefix = fresh(), fresh()
            try:
                def calculate():
                    output = self.bundle.model(self.mx.array([ids]), cache=full.entries)
                    short = self.bundle.model(self.mx.array([ids[:3]]), cache=prefix.entries)
                    self.mx.eval(output.logits, short.logits)
                    return error_metrics(np.asarray(output.logits[0, 2].astype(self.mx.float32)),
                        np.asarray(short.logits[0, -1].astype(self.mx.float32)))
                metric = self.engine._device.call(id(self.engine), calculate)
                assert metric["within_tolerance"], metric
                return metric
            finally:
                full.release()
                prefix.release()

        self.check("D06", masks)

        def isolated():
            prompts = [self.prompt(CORPUS[0]), self.prompt(CORPUS[1])]
            results = [self.generate(p, max_tokens=8)[0] for p in prompts]
            repeated = [self.generate(p, max_tokens=8)[0] for p in reversed(prompts)]
            assert [r["observer"]["token_ids"] for r in results] == [
                r["observer"]["token_ids"] for r in reversed(repeated)]
            return {"independent": results, "interleaved": repeated}

        for case in ("D07", "D08"):
            self.check(case, isolated)

        def clone():
            original, cloned = fresh(), None
            try:
                ids = self.bundle.tokenizer.encode("A shared prefix.")
                self.engine._device.call(id(self.engine), lambda: forward(original, ids))
                cloned = owned_clone(self.engine._device, id(self.engine), original)
                metrics = []
                for branch, suffix in ((original, " Different suffix."),
                                       (cloned, " Independent fork.")):
                    tail = self.bundle.tokenizer.encode(suffix)
                    oracle = fresh()
                    try:
                        a = self.engine._device.call(id(self.engine),
                            lambda c=branch, t=tail: forward(c, t))
                        b = self.engine._device.call(id(self.engine),
                            lambda c=oracle, t=tail: forward(c, ids + t))
                        metric = error_metrics(a, b)
                        assert metric["within_tolerance"], metric
                        metrics.append(metric)
                    finally:
                        oracle.release()
                return metrics
            finally:
                original.release()
                if cloned:
                    cloned.release()

        self.check("D09", clone)
        self.check("D10", lambda: self.require(component_comparisons and all(
            row["within_tolerance"] for row in component_comparisons), component_comparisons))

        def dtype():
            assert engine_observer and engine_observer.states
            states = engine_observer.states[-1]["components"]
            for component in states:
                for name, field in component["fields"].items():
                    if name in {"keys", "_keys", "_values"} or (
                            name == "values" and len(field["shape"]) == 4):
                        assert field["dtype"] == "float16"
                    if name == "recurrent_state":
                        assert field["dtype"] == "float32"
            return [{**s, "fields": {k: {a: b for a, b in v.items() if a != "array"}
                                     for k, v in s["fields"].items()}} for s in states]

        self.check("E01", dtype)
        for case in ("E02", "E03", "E04"):
            self.check(case, lambda: self.require(diagnostic["status"] == "PASS",
                                                {"positions": offsets}))
        for case in ("E05", "E06"):
            self.check(case, isolated)

        def cleanup():
            stats = self.engine._cache_manager.stats()
            assert stats.active_handles == stats.reserved_bytes == 0
            return stats.to_dict()

        def eviction():
            manager = self.engine._cache_manager
            handle = manager.create_handle("eviction-test")
            manager.release(handle)
            manager.clear_prefixes()
            return cleanup()

        self.check("E07", eviction)
        self.check("E08", lambda: self.cancel_at("decode"))
        self.check("E09", self.reload)

        def identity():
            manager = self.engine._cache_manager
            assert manager.fingerprint(namespace="principal-a") != manager.fingerprint(
                namespace="principal-b")
            return {"namespace_fingerprints_differ": True, "prefix_reuse_enabled": False}

        self.check("E10", identity)

        def restoration():
            if self.bundle.architecture in {"lfm2", "qwen3_5"}:
                raise NotImplementedError("dense hybrid prefix restoration is outside this scope")
            from mlx_one.p0_qualification import dense_snapshot_parity

            ids = self.bundle.tokenizer.encode("Shared prefix. Independent restored continuation.")
            result = self.engine._device.call(id(self.engine), lambda: dense_snapshot_parity(
                self.bundle, ids, max(1, len(ids) // 2), steps=3))
            assert result["passed"], "optional dense restoration parity failed"
            return result

        self.check("P01", restoration, applicable=False)

    def cancel_at(self, phase):
        ready, resume = threading.Event(), threading.Event()
        observed = Observer()

        def observer(event):
            observed(event)
            if event["kind"] == "forward" and event["phase"] == phase:
                ready.set()
                assert resume.wait(30), "cancellation boundary was not released"

        with observe_request(observer):
            handle = self.engine.submit(self.prompt(CORPUS[0]), max_tokens=128)
        try:
            assert ready.wait(30), f"{phase} boundary not reached"
            start = time.perf_counter()
            handle.cancel()
            handle.cancel()
        finally:
            resume.set()
        value = handle.result(timeout=30)
        stats = self.engine._cache_manager.stats()
        assert stats.active_handles == stats.reserved_bytes == 0
        return {"phase": phase, "completion_ms": (time.perf_counter() - start) * 1000,
                "result": value.to_dict(), "observation": observed.summary()}

    def reload(self):
        from mlx_one import Engine

        prompt = self.prompt(CORPUS[0])
        if not self.engine._closed:
            first, _ = self.generate(prompt, max_tokens=16)
            self.reload_expected = first
        else:
            first = self.reload_expected
        config = self.engine._cache_manager.config
        self.engine.close()
        old_manager = self.engine._cache_manager
        assert old_manager.stats().active_handles == old_manager.stats().reserved_bytes == 0
        try:
            old_manager.create_handle("stale")
        except RuntimeError:
            pass
        else:
            raise AssertionError("disposed manager accepted a stale request")
        old_manager = None
        self.bundle = None
        self.engine = None
        gc.collect()
        self.mx.clear_cache()
        self.engine = Engine(self.metadata["snapshot_path"], offline=True,
            revision=self.metadata["revision"], inference_mode="reference_dense",
            execution_dtype="float16", cache_config=config)
        self.bundle = self.engine.model
        second, _ = self.generate(prompt, max_tokens=16)
        assert first["observer"]["token_ids"] == second["observer"]["token_ids"]
        return {"first": first, "reloaded": second}

    def lifecycle(self):
        prompt = self.prompt(CORPUS[0])
        self.reload_expected = self.generate(prompt, max_tokens=16)[0]

        def sequential():
            runs = [self.engine.submit(prompt, max_tokens=8).result(timeout=120).to_dict()
                    for _ in range(20)]
            assert all(r["text"] == runs[0]["text"] for r in runs)
            return runs

        self.check("F01", sequential, seconds=600)

        def concurrent(count):
            async def run():
                return await asyncio.gather(*(self.engine.async_generate(
                    self.prompt(CORPUS[i]), max_tokens=16, temperature=.7, seed=44)
                    for i in range(count)))
            results = asyncio.run(run())
            independent = [self.engine.generate(self.prompt(CORPUS[i]), max_tokens=16,
                                               temperature=.7, seed=44) for i in range(count)]
            assert [r.text for r in results] == [r.text for r in independent]
            return {"requests": count, "results": [r.to_dict() for r in results]}

        for case, count in (("F02", 2), ("F03", 4), ("F04", 8)):
            self.check(case, lambda n=count: concurrent(n), seconds=600)

        def queued_cancel():
            ready, resume = threading.Event(), threading.Event()

            def observer(event):
                if event["kind"] == "forward":
                    ready.set()
                    assert resume.wait(30)

            with observe_request(observer):
                first = self.engine.submit(prompt, max_tokens=8)
            try:
                assert ready.wait(30)
                second = self.engine.submit(prompt, max_tokens=8)
                assert second.cancel()
                from concurrent.futures import CancelledError
                try:
                    second.result(timeout=2)
                except CancelledError:
                    pass
                else:
                    raise AssertionError("queued cancellation did not cancel future")
            finally:
                resume.set()
            first.result(timeout=30)
            return {"queued_cancelled": True, "peer_completed": True}

        self.check("F05", queued_cancel)
        self.check("F06", lambda: self.cancel_at("prefill"))
        self.check("F07", lambda: self.cancel_at("decode"))

        def exception():
            async def invalid():
                try:
                    await self.engine.async_generate("", max_tokens=8)
                except ValueError:
                    return {"exception_propagated": True}
                raise AssertionError("async invalid prompt did not propagate")
            return asyncio.run(invalid())

        self.check("F08", exception)

        def async_cancel():
            ready, resume = threading.Event(), threading.Event()
            forwards = []

            def observer(event):
                if event["kind"] == "forward":
                    forwards.append(event["phase"])
                    ready.set()
                    assert resume.wait(30)

            async def exercise():
                with observe_request(observer):
                    task = asyncio.create_task(self.engine.async_generate(prompt, max_tokens=128))
                assert await asyncio.to_thread(ready.wait, 30)
                task.cancel()
                resume.set()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("async cancellation did not propagate")
                stop = time.monotonic() + 30
                while self.engine._cache_manager.stats().active_handles:
                    assert time.monotonic() < stop, "async cancellation left a cache handle"
                    await asyncio.sleep(.01)
                assert len(forwards) <= 1, "cancelled async request continued model execution"
                return {"native_forwards": len(forwards), "handles_after_cancel": 0}

            try:
                return asyncio.run(exercise())
            finally:
                resume.set()

        self.check("F11", async_cancel)

        def active_close():
            ready, resume = threading.Event(), threading.Event()

            def observer(event):
                if event["kind"] == "forward":
                    ready.set()
                    assert resume.wait(30)

            with observe_request(observer):
                handle = self.engine.submit(prompt, max_tokens=64)
            assert ready.wait(30), "request did not reach an actual forward"
            closer = threading.Thread(target=self.engine.close)
            closer.start()
            try:
                stop = time.monotonic() + 10
                while not self.engine._closed and time.monotonic() < stop:
                    time.sleep(.01)
                assert self.engine._closed
            finally:
                resume.set()
            closer.join(30)
            assert not closer.is_alive(), "close with active request hung"
            # Running requests may complete or cooperatively cancel, never hang or leak.
            from concurrent.futures import CancelledError
            try:
                handle.result(timeout=30)
            except CancelledError:
                pass
            assert self.engine._cache_manager.stats().active_handles == 0
            return {"closed": True, "handle_done": handle.done()}

        self.check("F09", active_close)
        self.check("F10", self.reload)

        def direct_stream_close():
            observer = Observer()
            with observe_request(observer):
                stream = self.engine.stream(prompt, max_tokens=16)
                next(stream)
                self.engine.close()
                tail = list(stream)
            assert not any(chunk.token_id is not None for chunk in tail)
            assert len(observer.batch_sizes) == 1
            assert self.engine._cache_manager.stats().active_handles == 0
            self.reload()
            return {"forwards": 1, "post_close_tokens": 0, "reload_verified": True}

        self.check("F12", direct_stream_close)
        for case in ("G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G10"):
            functions = {"G01": lambda: self.engine.capabilities(),
                         "G02": lambda: self.generate(prompt, max_tokens=16)[0],
                         "G03": lambda: self.generate(prompt, stream=True, max_tokens=16)[0],
                         "G04": lambda: self.cancel_at("decode"),
                         "G05": self.reload, "G06": self.reload, "G07": self.reload,
                         "G08": lambda: self.require(
                             self.engine._cache_manager.stats().active_handles == 0 and
                             self.engine._cache_manager.stats().reserved_bytes == 0),
                         "G10": self.reload}
            self.check(case, functions[case])
        self.check("G09", self.cycles, seconds=3600)

    def cycles(self):
        from mlx_one.engine.pressure import sample_telemetry

        samples = []
        # Exercise repeated allocation/release on one live model before reload testing.
        for _ in range(100):
            self.generate(self.prompt(CORPUS[0]), max_tokens=8)
        for index in range(20):
            self.reload()
            for _ in range(5):
                self.generate(self.prompt(CORPUS[0]), max_tokens=8)
            self.mx.synchronize()
            gc.collect()
            self.mx.clear_cache()
            telemetry = sample_telemetry()
            stats = self.engine._cache_manager.stats()
            assert stats.active_handles == stats.reserved_bytes == 0
            samples.append({"cycle": index, "rss_bytes": telemetry.rss_bytes,
                "mlx_active_bytes": self.mx.get_active_memory(),
                "mlx_cached_bytes": self.mx.get_cache_memory(),
                "peak_bytes": self.mx.get_peak_memory(), "cache": stats.to_dict(),
                "device_queue": self.engine._device.stats()["queued"],
                "request_queue": self.engine._executor._work_queue.qsize()})
            assert samples[-1]["device_queue"] == samples[-1]["request_queue"] == 0
            self.rows["memory"] = [{"model_id": self.metadata["model_id"], **s} for s in samples]
            self.save()
        warmed = samples[3:]
        growth = {}
        for metric, limit in (("rss_bytes", 128 * 1024**2),
                              ("mlx_active_bytes", 32 * 1024**2)):
            assert all(s[metric] is not None for s in warmed), "missing memory telemetry"
            delta = statistics.median(s[metric] for s in warmed[-3:]) - statistics.median(
                s[metric] for s in warmed[:3])
            growth[metric] = {"growth_bytes": delta, "limit_bytes": limit}
            assert delta <= limit, growth
        return {"load_close_cycles": 20, "generation_cycles": 200,
                "loaded_generation_cycles": 100, "reload_generation_cycles": 100,
                "warmup_cycles_discarded": 3, "growth": growth}

    def benchmark(self, lengths=None):
        from mlx_one.engine.pressure import sample_telemetry
        for length in lengths or self.manifest["environment"]["policy"].get(
                "benchmark_prompt_tokens", (128, 512, 2048, 8192)):
            applicable = length + 128 <= self.bundle.context_length

            def measure(n=length):
                if n + 128 > self.bundle.context_length:
                    raise NotImplementedError("prompt plus output exceeds checkpoint context")
                prompt = exact_prompt(self.bundle.tokenizer, n)
                assert len(self.bundle.tokenizer.encode(prompt)) == n
                prompt_path = self.directory / f"prompt-{n}.json"
                atomic_json(prompt_path, {"text": prompt,
                    "token_ids": self.bundle.tokenizer.encode(prompt),
                    "tokenizer_identity": self.bundle.tokenizer_identity})
                prompt_digest = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
                self.generate(prompt, max_tokens=128, admission=True)
                records = []
                for repetition in range(5):
                    self.mx.synchronize()
                    self.mx.reset_peak_memory()
                    started_monotonic = time.monotonic()
                    started = time.perf_counter()
                    result, observer = self.generate(prompt, max_tokens=128, admission=True)
                    self.mx.synchronize()
                    finished_monotonic = time.monotonic()
                    synchronized_wall = time.perf_counter() - started
                    elapsed = result["engine_wall_seconds"]
                    timing = observer.summary()
                    telemetry = sample_telemetry()
                    row = {"model_id": self.metadata["model_id"],
                        "revision": self.metadata["revision"], "prompt_tokens": n,
                        "generated_tokens": result["generation_tokens"], "repetition": repetition,
                        "mode": "reference_dense", "batch": 1, "execution_dtype": "float16",
                        "kv_dtype": "float16", "prefix_reuse": False, "prefill_chunk_tokens": None,
                        "cold_worker_load_seconds": self.load_seconds,
                        "case_id": f"H{n}", "timestamp": utc(),
                        "source": self.manifest["environment"]["source"],
                        "started_monotonic": started_monotonic,
                        "finished_monotonic": finished_monotonic,
                        "end_to_end_seconds": elapsed,
                        "harness_synchronized_wall_seconds": synchronized_wall,
                        "rendered_prompt_artifact": str(prompt_path),
                        "rendered_prompt_sha256": prompt_digest,
                        "end_to_end_tokens_per_second": result["generation_tokens"] / elapsed,
                        "prefill_ms": timing["forward_ms"]["prefill"],
                        "prefill_tokens_per_second": n / (timing["forward_ms"]["prefill"] / 1000),
                        "decode_forward_ms": timing["forward_ms"]["decode"],
                        "rss_bytes": telemetry.rss_bytes,
                        "mlx_active_bytes": self.mx.get_active_memory(),
                        "mlx_cached_bytes": self.mx.get_cache_memory(),
                        "mlx_peak_bytes": self.mx.get_peak_memory(), **timing}
                    assert timing["actual_prefill_tokens"] == n
                    assert timing["batch_sizes"] and set(timing["batch_sizes"]) == {1}
                    records.append(row)
                    self.rows["benchmarks"].append(row)
                    self.save()
                return records

            self.check(f"H{length}", measure, seconds=1800, applicable=applicable)

    def _ownership_snapshot(self, label):
        from dataclasses import asdict

        from mlx_one.engine.pressure import sample_telemetry

        manager = self.engine._cache_manager
        stats = manager.stats()
        assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0
        assert not manager._handles and not manager._reservations
        assert stats.prefix_bytes == 0  # Mandatory baseline has optional prefix reuse disabled.
        record = {'label': label, 'cache': stats.to_dict(),
                  'telemetry': asdict(sample_telemetry())}
        self.rows.setdefault('ownership', []).append(record)
        self.save()
        return record

    def admission(self):
        if 8192 + 128 > self.bundle.context_length:
            for index in range(1, 6):
                self.check(f'I{index:02}', lambda: (_ for _ in ()).throw(
                    NotImplementedError('8K exceeds checkpoint context')), applicable=False)
            return
        prompt = exact_prompt(self.bundle.tokenizer, 8192)
        expected = None

        def request(label):
            value, _ = self.generate(prompt, max_tokens=128, admission=True)
            assert value['observer']['actual_prefill_tokens'] == 8192
            assert set(value['observer']['batch_sizes']) == {1}
            self._ownership_snapshot(label)
            if expected is not None:
                assert value['observer']['token_ids'] == expected
            return value

        cold = self.check('I01', lambda: request('cold-8192'), seconds=600)
        if cold['status'] == 'PASS':
            expected = cold['details']['observer']['token_ids']

        def warm():
            for length in (128, 512, 2048):
                for repetition in range(6):
                    self.generate(exact_prompt(self.bundle.tokenizer, length),
                                  max_tokens=128, admission=True)
                    self._ownership_snapshot(f'warm-{length}-{repetition}')
            return request('warm-8192')
        self.check('I02', warm, seconds=3600)

        def cancelled():
            cancellation = self.cancel_at('decode')
            self._ownership_snapshot('after-cancellation')
            return {'cancellation': cancellation, 'request': request('post-cancel-8192')}
        self.check('I03', cancelled, seconds=900)

        def cleaned():
            before = self._ownership_snapshot('before-explicit-cleanup')
            def reclaim():
                assert self.engine._device.is_owner
                manager = self.engine._cache_manager
                assert manager.stats().active_handles == manager.stats().reserved_bytes == 0
                self.mx.synchronize()
                manager.clear_prefixes()
                manager.block_pool.discard_free()
                self.mx.clear_cache()
                manager._capacity_snapshot = None
            self.engine._device.call(id(self.engine), reclaim)
            after = self._ownership_snapshot('after-explicit-cleanup')
            return {'before': before, 'after': after, 'request': request('post-cleanup-8192')}
        self.check('I04', cleaned, seconds=900)

        def repeated():
            values = [request('repeated-8192-' + str(index)) for index in range(2)]
            assert expected is not None
            assert all(v['observer']['token_ids'] == expected for v in values)
            return {'repetitions': values, 'greedy_agreement': True}
        self.check('I05', repeated, seconds=1800)

    def admission_independent(self):
        def independent():
            if 8192 + 128 > self.bundle.context_length:
                raise NotImplementedError('8K exceeds checkpoint context')
            previous = json.loads((self.directory.parent / 'admission/results.json').read_text())
            cold = next(r for r in previous['cases'] if r['case_id'] == 'I01')
            assert cold['status'] == 'PASS', 'independent comparison requires cold success'
            value, _ = self.generate(exact_prompt(self.bundle.tokenizer, 8192),
                                     max_tokens=128, admission=True)
            assert value['observer']['token_ids'] == cold['details']['observer']['token_ids']
            return {'request': value, 'independent_fresh_process': True,
                    'ownership': self._ownership_snapshot('independent-worker')}
        self.check('I06', independent, seconds=600,
                   applicable=8192 + 128 <= self.bundle.context_length)

    def close(self):
        self.engine.close()
        self.engine.close()
        stats = self.engine._cache_manager.stats()
        assert stats.active_handles == stats.reserved_bytes == 0
        return {"repeated_close": True, "cache": stats.to_dict()}


def main():
    manifest_path, suite, destination = sys.argv[1:]
    directory = Path(destination)
    manifest = json.loads(Path(manifest_path).read_text())
    cases = None
    try:
        cases = Cases(manifest, directory)
        getattr(cases, suite)()
        return 0
    except BaseException as exc:
        atomic_json(directory / "fatal.json", {
            "error": str(exc), "traceback": traceback.format_exc()})
        if cases is None:
            atomic_json(directory / "results.json", {"cases": [{"case_id": "A03",
                "status": "FAIL", "error_type": type(exc).__name__, "reason": str(exc),
                "traceback": traceback.format_exc()}]})
        return 1
    finally:
        if cases is not None:
            try:
                cases.close()
            except Exception as exc:
                atomic_json(directory / "cleanup-failure.json", {"error": str(exc)})
                cases.rows["cases"].append({"case_id": "G08", "status": "FAIL",
                    "reason": str(exc), "shared_blocker": True, "timestamp": utc()})
            cases.save()


if __name__ == "__main__":
    raise SystemExit(main())
