"""Isolated, local-only HF oracle for the Batch 1 BERT retrieval profiles.

Run only in the reference environment. Torch/Transformers are development tools,
never dependencies of native inference. No code or weights are fetched here.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nomic-reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    spec = json.loads(args.input.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    rerank = spec["task"] == "reranking"
    is_nomic = json.loads((args.model / "config.json").read_text())["model_type"] == "nomic_bert"
    captured = {}
    if is_nomic:
        if args.nomic_reference is None:
            raise ValueError("Nomic requires the inspected, pinned local reference source")
        package = types.ModuleType("batch1_nomic_reference")
        package.__path__ = [str(args.nomic_reference)]
        sys.modules[package.__name__] = package
        modules = {}
        for name in ("configuration_hf_nomic_bert", "modeling_hf_nomic_bert"):
            spec_module = importlib.util.spec_from_file_location(
                package.__name__ + "." + name, args.nomic_reference / (name + ".py")
            )
            module = importlib.util.module_from_spec(spec_module)
            sys.modules[spec_module.name] = module
            spec_module.loader.exec_module(module)
            modules[name] = module
        config = modules["configuration_hf_nomic_bert"].NomicBertConfig.from_dict(
            json.loads((args.model / "config.json").read_text())
        )
        model = modules["modeling_hf_nomic_bert"].NomicBertModel(config, add_pooling_layer=False)
        from safetensors.torch import load_file

        model.load_state_dict(load_file(args.model / "model.safetensors"), strict=True)
        model.emb_ln.register_forward_hook(lambda m, i, o: captured.update(embedding_hidden=o))
        model.encoder.layers[0].register_forward_hook(
            lambda m, i, o: captured.update(first_layer=o[0])
        )
    else:
        factory = AutoModelForSequenceClassification if rerank else AutoModel
        options = {"add_pooling_layer": False} if spec.get("bge_m3") else {}
        model = factory.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False, **options
        )
    model.eval()
    inputs = tokenizer(
        spec["first"],
        spec["second"] if rerank else None,
        padding=True,
        truncation=True,
        max_length=spec["max_length"],
        return_tensors="pt",
    )
    arrays = {name: value.numpy() for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs) if is_nomic else model(**inputs, output_hidden_states=True)
    arrays["embedding_hidden"] = (
        captured["embedding_hidden"].numpy() if is_nomic else output.hidden_states[0].numpy()
    )
    arrays["first_layer"] = (
        captured["first_layer"].numpy() if is_nomic else output.hidden_states[1].numpy()
    )
    arrays["hidden"] = (
        output.last_hidden_state.numpy() if is_nomic else output.hidden_states[-1].numpy()
    )
    if rerank:
        arrays["output"] = output.logits.numpy()
    else:
        hidden = output.last_hidden_state
        if spec["pooling"] == "cls":
            vectors = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1)
            vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        if spec["normalize"]:
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        arrays["output"] = vectors.numpy()
        if spec.get("bge_m3"):
            sparse = torch.load(
                args.model / "sparse_linear.pt", map_location="cpu", weights_only=True
            )
            colbert = torch.load(
                args.model / "colbert_linear.pt", map_location="cpu", weights_only=True
            )
            # Publisher auxiliary files are FP16-stored; use_fp16=False in the
            # publisher pipeline promotes the entire model, including its heads.
            sparse = {key: value.to(hidden.dtype) for key, value in sparse.items()}
            colbert = {key: value.to(hidden.dtype) for key, value in colbert.items()}
            arrays["sparse_weights"] = (
                torch.relu(
                    torch.nn.functional.linear(hidden, sparse["weight"], sparse["bias"])[..., 0]
                )
                * inputs["attention_mask"]
            ).numpy()
            token_vectors = torch.nn.functional.linear(
                hidden[:, 1:], colbert["weight"], colbert["bias"]
            )
            arrays["colbert_embeddings"] = (
                torch.nn.functional.normalize(token_vectors, p=2, dim=-1)
                * inputs["attention_mask"][:, 1:, None]
            ).numpy()
        if is_nomic:
            for dim in (64, 128, 256, 512, 768):
                normalized = torch.nn.functional.layer_norm(
                    (hidden * mask).sum(1) / mask.sum(1).clamp(min=1), (hidden.shape[-1],)
                )
                arrays[f"matryoshka_{dim}"] = torch.nn.functional.normalize(
                    normalized[:, :dim], p=2, dim=1
                ).numpy()
    np.savez_compressed(args.output, **arrays)
    source = Path(inspect.getfile(type(model)))
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "source_module": type(model).__module__,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "parameter_count": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
