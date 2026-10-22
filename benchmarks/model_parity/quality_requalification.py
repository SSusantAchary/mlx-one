"""Task-budget requalification using unchanged prompts and exact expected answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from four_path_oracle import load_reference
from quality import evaluate


def run(path, precision, output):
    import mlx.core as mx

    from mlx_one import Engine
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def prepare():
        return load_reference(bundle, precision)

    bundle, reference, _ = manager._pool.executor.call(bundle.model_id, prepare)
    fixture = (
        Path(__file__).resolve().parents[2] / "scripts/qualification/ten_model_quality_v2.json"
    )
    report = {
        "checkpoint": bundle.model_id,
        "revision": bundle.revision,
        "precision": precision,
        "qualification": "NOT_QUALIFIED",
    }
    backbone = (
        bundle.model.transformer
        if bundle.architecture == "gpt2"
        else bundle.model.language_model
        if bundle.architecture == "qwen3_5"
        else bundle.model.model
    )
    embedding = backbone.wte if bundle.architecture == "gpt2" else backbone.embed_tokens
    activation_dtype = (
        embedding.scales.dtype if hasattr(embedding, "scales") else embedding.weight.dtype
    )
    storage_dtype = mx.bfloat16 if activation_dtype == mx.bfloat16 else mx.float16
    cache_type = "bf16" if storage_dtype == mx.bfloat16 else "f16"
    report["cache_profile"] = {
        "native_kv": str(storage_dtype),
        "reference_kv": str(storage_dtype),
        "activation": str(activation_dtype),
        "note": "The public Engine supports FP16/BF16 KV; GPT-2 FP32 uses explicitly matched "
        "FP16 storage in this task comparator. Numerical raw-cache gates remain separate.",
    }

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    class ReferenceEngine:
        def stream(self, prompt, *, options):
            def generate():
                from local_reference import reference_module

                ids = bundle.tokenizer.encode(prompt)
                cache = reference_module("cache").make_prompt_cache(reference)

                class StorageView:
                    def __init__(self, owner):
                        self.owner = owner

                    def __getattr__(self, name):
                        return getattr(self.owner, name)

                    def update_and_fetch(self, keys, values):
                        dtype = keys.dtype
                        k, v = self.owner.update_and_fetch(
                            keys.astype(storage_dtype), values.astype(storage_dtype)
                        )
                        return k.astype(dtype), v.astype(dtype)

                cache = [StorageView(c) if hasattr(c, "update_and_fetch") else c for c in cache]
                logits = reference(mx.array([ids]), cache=cache)
                generated, finish = [], "length"
                for step in range(options.max_tokens):
                    token = int(mx.argmax(logits[0, -1]).item())
                    if token in bundle.eos_token_ids:
                        finish = "stop"
                        break
                    generated.append(token)
                    if step + 1 < options.max_tokens:
                        logits = reference(mx.array([[token]]), cache=cache)
                text = bundle.tokenizer.decode(generated)
                return [
                    *[SimpleNamespace(text="", token_id=i, finish_reason=None) for i in generated],
                    SimpleNamespace(text=text, token_id=None, finish_reason=finish),
                ]

            return iter(manager._pool.executor.call(bundle.model_id, generate))

    try:
        execution = "float16" if precision.startswith("affine_q") else None
        with Engine(
            bundle,
            execution_dtype=execution,
            inference_mode="reference_dense",
            cache_config=CacheConfig(
                prefix_cache_enabled=False, key_type=cache_type, value_type=cache_type
            ),
        ) as engine:
            report["native"] = evaluate(
                bundle, engine, fixture, chat_applicable=bundle.architecture != "gpt2"
            )
            report["cleanup"] = engine.runtime_stats()["cache_stats"]["active_handles"] == 0
            save()
        report["reference"] = evaluate(
            bundle, ReferenceEngine(), fixture, chat_applicable=bundle.architecture != "gpt2"
        )
        report["reference_contract"] = (
            "Independent greedy cached generation with identical prompt IDs, source transform, "
            "EOS IDs, budgets and KV storage dtype. A reference cache view casts storage and "
            "restores activation dtype at the attention boundary, matching the public native "
            "cache boundary. Recurrent state remains architecture-owned. This is task evidence, "
            "not a substitute for unmodified-reference raw-cache mathematical gates."
        )
        report["status"] = report["native"]["status"]
        save()
        print(
            bundle.model_id,
            precision,
            report["native"]["passed"],
            report["native"]["scored"],
            flush=True,
        )
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.checkpoint, args.precision, args.output)


if __name__ == "__main__":
    main()
