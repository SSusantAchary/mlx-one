"""Matched-MLX pinned-source generation and optional cached LFM2 Q4 baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from qualify import REFERENCE  # supplies Git-tree generated version metadata only

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen2", "qwen3", "smollm2", "lfm2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    import numpy as np
    from evidence import snapshot
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.text import load_text_model

    models = json.loads((ROOT / "local/competitor-comparison/models.json").read_text())["models"]
    if args.model == "lfm2":
        path = (
            Path.home()
            / ".cache/huggingface/hub/models--mlx-community--LFM2-350M-4bit"
            / "snapshots/18dc72abf3b2337f9123cfd6eeeb58dfa7947066"
        )
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError("verified cached Q4 checkpoint unavailable; no download")
        spec = {
            "key": "lfm2",
            "path": str(path),
            "revision": path.name,
            "execution_dtype": "float16",
            "files_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in path.iterdir()
                if p.is_file() and p.suffix in {".json", ".safetensors"}
            },
        }
        native = load_text_model(path, offline=True)
        ids = native.tokenizer.encode(
            "A research team explored the old city and kept notes. " * 32
        )[:128]
        spec["prompts"] = [{"token_ids": ids}]
        configure_execution_dtype(native, "float16")
    else:
        spec = next(m for m in models if m["key"] == args.model)
        native = None
    snapshot(args.output, model=spec)
    reference, config = load_model(Path(spec["path"]))
    reference.set_dtype(
        getattr(mx, spec["execution_dtype"]),
        predicate=lambda d: d in {mx.float16, mx.bfloat16, mx.float32},
    )
    mx.eval(reference.parameters())
    eos = config.get("eos_token_id", [])
    eos = {eos} if isinstance(eos, int) else set(eos)
    results = {
        "model": spec,
        "source_tree": str(REFERENCE),
        "mlx_version": importlib.metadata.version("mlx"),
        "runs": [],
        "scope": "matched dependency native reference diagnostic; not original HTTP",
    }

    for prompt_index in [0] if native else [1, 2]:
        ids = spec["prompts"][prompt_index]["token_ids"]
        variants = ["full_prefill_argmax", "split_prefill_argmax", "pinned_generate_step"]
        if native is not None:
            variants.append("native_full_prefill_argmax")
        for variant in variants:
            for repetition in range(3):
                tokens, terminal = [], None
                mx.synchronize()
                started = time.perf_counter()
                if variant == "pinned_generate_step":
                    iterator = generate_step(
                        mx.array(ids), reference, max_tokens=128, prefill_step_size=512
                    )
                    try:
                        for token, _ in iterator:
                            if token in eos:
                                terminal = token
                                break
                            tokens.append(token)
                    finally:
                        iterator.close()
                else:
                    is_native = variant == "native_full_prefill_argmax"
                    cache = native.model.make_cache() if is_native else make_prompt_cache(reference)
                    current = ids
                    if variant.startswith("split"):
                        reference(mx.array([ids[:-1]]), cache=cache)
                        mx.eval([c.state for c in cache])
                        current = ids[-1:]
                    for _ in range(128):
                        logits = (
                            native.model(mx.array([current]), cache=cache).logits
                            if is_native
                            else reference(mx.array([current]), cache=cache)
                        )
                        token = int(mx.argmax(logits[0, -1]).item())
                        if token in eos:
                            terminal = token
                            break
                        tokens.append(token)
                        current = [token]
                mx.synchronize()
                elapsed = time.perf_counter() - started
                results["runs"].append(
                    {
                        "variant": variant,
                        "prompt_index": prompt_index,
                        "warmup": repetition == 0,
                        "tokens": tokens,
                        "terminal": terminal,
                        "finish_reason": "stop" if terminal is not None else "length",
                        "wall_seconds": elapsed,
                        "tokens_per_second": len(tokens) / elapsed,
                    }
                )
                print(
                    args.model,
                    prompt_index,
                    variant,
                    len(tokens),
                    len(tokens) / elapsed,
                    flush=True,
                )
    if native is not None:
        a, b = make_prompt_cache(reference), native.model.make_cache()
        current, comparisons = ids, []
        for step in range(128):
            x, y = (
                reference(mx.array([current]), cache=a),
                native.model(mx.array([current]), cache=b).logits,
            )
            mx.eval(x, y)
            next_token = int(mx.argmax(x[0, -1]).item())
            pairs = [("logits", x[:, -1:], y[:, -1:])]
            if step in {0, 15, 127}:
                for layer, (left, right) in enumerate(zip(a, b, strict=True)):
                    if callable(getattr(right, "state", None)):
                        pairs.extend(
                            (f"{layer}.{kind}", u[:, :, : left.offset], v)
                            for kind, u, v in zip(
                                ["K", "V"], left.state[:2], right.state(), strict=True
                            )
                        )
                    else:
                        pairs.append((f"{layer}.conv_carry", left[0], right.values))
            for name, x, y in pairs:
                comparisons.append(
                    {
                        "step": step,
                        "name": name,
                        **compare_tensors(
                            np.array(x.astype(mx.float32)),
                            np.array(y.astype(mx.float32)),
                            dtype="float16",
                        ),
                    }
                )
            current = [next_token]
        results["source_state_comparisons"] = comparisons
        results["source_state_parity"] = all(c["status"] == "PASS" for c in comparisons)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    snapshot(args.output, model=spec, status="MEASURED")


if __name__ == "__main__":
    main()
