"""Tiny native checkpoints for Metal integration and installed-wheel smoke tests."""

import json
from dataclasses import asdict
from pathlib import Path


def create_checkpoint(root, family="qwen2_vl", bits=4, liquid_variant=False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from test_native_lfm_mlx import dense_config
    from test_native_qwen_mlx import vl_config
    from test_native_qwen_multimodal_mlx import qwen2_5_vl_config, qwen3_5_config
    from tokenizers import Tokenizer, models, pre_tokenizers

    from mlx_one.core.registry import get_registration

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if family == "lfm2_vl":
        payload = {
            "model_type": "lfm2-vl",
            "image_token_index": 60,
            "text_config": asdict(
                dense_config(
                    max_position_embeddings=512, intermediate_size=64, block_multiple_of=32
                )
            ),
            "vision_config": {
                "model_type": "siglip2_vision_model",
                "hidden_size": 12,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "patch_size": 2,
                "num_patches": 4,
                "hidden_act": "gelu_pytorch_tanh",
            },
            "downsample_factor": 2,
            "projector_hidden_size": 16,
            "projector_use_layernorm": not liquid_variant,
        }
    else:
        config = {
            "qwen2_vl": vl_config,
            "qwen2_5_vl": qwen2_5_vl_config,
            "qwen3_5": qwen3_5_config,
        }[family]()
        payload = asdict(config)
        payload["text_config"]["max_position_embeddings"] = 512
    payload["eos_token_id"] = [2, 59]
    registration = get_registration(family)
    config = registration.config_class().from_dict(payload)
    model = registration.model_class()(config)
    nn.quantize(
        model,
        group_size=32,
        bits=bits,
        class_predicate=lambda path, module: (
            path.startswith("language_model.")
            and isinstance(module, nn.Linear)
            and module.weight.shape[-1] % 32 == 0
        ),
    )
    tensors = dict(tree_flatten(model.parameters()))
    assert any(n.endswith(".scales") for n in tensors)
    payload["quantization"] = {"bits": bits, "group_size": 32}
    published = {}
    for name, value in tensors.items():
        if family in {"qwen2_vl", "qwen2_5_vl"}:
            name = (
                name.replace("language_model.", "language_model.model.", 1)
                if name.startswith("language_model.")
                else name
            )
            name = name.replace("visual.", "vision_tower.", 1)
            if name == "vision_tower.patch_embed.proj.weight":
                v = config.vision_config
                value = value.reshape(
                    value.shape[0], 3, v.temporal_patch_size, v.patch_size, v.patch_size
                ).transpose(0, 2, 3, 4, 1)
        elif family == "lfm2_vl":
            name = name.replace("vision_model.projector_norm.", "multi_modal_projector.layer_norm.")
            name = name.replace("vision_model.projector_1.", "multi_modal_projector.linear_1.")
            name = name.replace("vision_model.projector_2.", "multi_modal_projector.linear_2.")
            name = name.replace("vision_model.blocks.", "vision_tower.vision_encoder.layers.")
            name = name.replace(
                "vision_model.patch_embedding.", "vision_tower.vision_embeddings.patch_embedding."
            )
            name = name.replace(
                "vision_model.position_embedding.",
                "vision_tower.vision_embeddings.position_embedding.",
            )
            name = name.replace(
                "vision_model.post_layernorm.", "vision_tower.vision_post_layernorm."
            )
        published[name] = value
    mx.save_safetensors(str(root / "model.safetensors"), published)
    (root / "config.json").write_text(json.dumps(payload))
    metadata = {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "min_pixels": 16, "max_pixels": 64}
    if family == "lfm2_vl":
        metadata.update(
            image_processor_type="Lfm2VlNumpyImageProcessor",
            min_image_tokens=1,
            max_image_tokens=4,
            do_image_splitting=False,
            use_image_special_tokens=True,
        )
    (root / ("processor_config.json" if liquid_variant else "preprocessor_config.json")).write_text(
        json.dumps({"image_processor": metadata} if liquid_variant else metadata)
    )
    vocabulary = {f"w{i}": i for i in range(64)}
    for token, index in {
        "[UNK]": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<image>": 60,
        "<|vision_start|>": 62,
        "<|vision_end|>": 63,
        "<|image_start|>": 57,
        "<|image_end|>": 58,
    }.items():
        del vocabulary[f"w{index}"]
        vocabulary[token] = index
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(
        [
            "<image>",
            "<bos>",
            "<eos>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_start|>",
            "<|image_end|>",
        ]
    )
    tokenizer.save(str(root / "tokenizer.json"))
    template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }}\n{% endfor %}assistant "
    (root / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "chat_template": template})
    )
    return root
