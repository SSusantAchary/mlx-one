"""Bounded local-only functional smoke and metrics; downloads nothing.

Run one checkpoint at a time. A PASS establishes only the operations recorded,
not model quality, 128K context, hardware qualification or user validation.
"""

import argparse
import gc
import json
import threading
import time
from pathlib import Path


def run(path, *, prompt, chat=False, max_tokens=16, context=512):
    import mlx.core as mx

    from mlx_one.engine.device_executor import DeviceExecutor
    from mlx_one.server.generation_engine import GenerationEngine
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.text import TextGenerationOptions

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint must be a local directory")
    manager = ModelManager()
    engine = GenerationEngine(
        manager,
        inference_mode="reference_dense",
        collect_execution_metrics=True,
        prefix_cache_bytes=0,
    )
    started = time.perf_counter()
    report = {
        "model_path": str(root),
        "checkpoint_status": "MANUAL_TEST_PENDING",
        "scope": "bounded functional smoke, no quality or numerical research gate",
    }
    try:
        bundle = manager.load(root, offline=True)
        executor = DeviceExecutor.shared()
        executor.call(id(bundle), lambda: mx.eval(bundle.model.parameters()))
        report.update(
            load_seconds=time.perf_counter() - started,
            architecture=bundle.architecture,
            model_id=bundle.model_id,
            revision=bundle.revision,
            configured_maximum=bundle.context_length,
            smoke_context=context,
        )

        def continuation():
            ids = bundle.tokenizer.encode(prompt)
            ids = (ids * 3)[:3] if ids else [bundle.tokenizer.bos_token_id] * 3
            if any(token is None for token in ids):
                raise ValueError("prompt is empty and tokenizer has no BOS")
            cache = bundle.model.make_cache()
            try:
                a = bundle.model(mx.array([ids[:2]]), cache=cache).logits
                b = bundle.model(mx.array([ids[2:]]), cache=cache).logits
                mx.eval(a, b)
                if not mx.all(mx.isfinite(a)).item() or not mx.all(mx.isfinite(b)).item():
                    raise ValueError("nonfinite prefill/continuation logits")
                if any(state.offset != 3 for state in cache):
                    raise ValueError("cache positions did not advance")
                cache_bytes = sum(state.nbytes for state in cache)
                for state in cache:
                    state.reset()
                if any(state.offset != 0 for state in cache):
                    raise ValueError("cache positions did not reset")
                return {"status": "PASS", "tokens": 3, "state_bytes": cache_bytes}
            finally:
                for state in cache:
                    state.reset()

        report["cache_continuation_reset"] = executor.call(id(bundle), continuation)
        options = TextGenerationOptions(max_tokens=max_tokens, temperature=0)
        settings = dict(
            model=bundle.model_id,
            options=options,
            cancel=threading.Event(),
            context_length=min(context, bundle.context_length),
            cache_prompt=False,
        )
        executor.call(id(bundle), mx.reset_peak_memory)
        events = list(
            engine.stream(messages=[{"role": "user", "content": prompt}], **settings)
            if chat
            else engine.stream_completion(prompt=prompt, **settings)
        )
        report["text"] = "".join(event.text for event in events)
        report["metrics"] = engine.stats.to_dict()
        report["runtime"] = engine.runtime_stats()
        if not events or events[-1].finish_reason not in {"stop", "length"}:
            raise ValueError("generation did not finish")
        if not report["metrics"]["generated_tokens"]:
            raise ValueError("no output tokens generated; inspect EOS/template behavior")
        if chat and not report["text"].strip():
            raise ValueError("no visible final chat text within this output budget")
        report["checkpoint_status"] = "MODEL_SMOKE_TESTED"
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        engine.close()
        manager.unload()
        gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 128 <= args.context <= 1024 or not 1 <= args.max_tokens <= 128:
        parser.error("smoke context must be 128..1024 and output budget 1..128")
    if args.output.exists():
        parser.error("output already exists; use a new evidence filename")
    result = run(
        args.model,
        prompt=args.prompt,
        chat=args.chat,
        max_tokens=args.max_tokens,
        context=args.context,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "checkpoint_status", "text", "error")
                if key in result
            },
            indent=2,
        )
    )
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
