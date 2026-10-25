"""Tiny genuine native checkpoints; never download model weights."""

import json
from pathlib import Path


def tiny_config(family):
    text = dict(
        model_type="qwen2",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        eos_token_id=2,
    )
    if family == "trocr":
        return dict(
            model_type="vision-encoder-decoder",
            encoder=dict(
                model_type="vit",
                image_size=16,
                patch_size=4,
                num_channels=3,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=2,
                num_hidden_layers=1,
                layer_norm_eps=1e-6,
                qkv_bias=False,
            ),
            decoder=dict(
                model_type="trocr",
                d_model=16,
                decoder_attention_heads=2,
                decoder_layers=1,
                decoder_ffn_dim=32,
                max_position_embeddings=128,
                cross_attention_hidden_size=16,
                vocab_size=64,
                eos_token_id=2,
                decoder_start_token_id=2,
            ),
        )
    if family == "got_ocr2":
        return dict(
            **{**text, "model_type": "GOT"},
            im_patch_token=4,
            im_start_token=5,
            im_end_token=6,
            vision_config=dict(
                img_size=32,
                patch_size=4,
                embed_dim=16,
                depth=1,
                num_heads=2,
                window_size=2,
                global_attn_indexes=[0],
                out_chans=8,
                out_dim=16,
            ),
        )
    if family == "florence2":
        return dict(
            model_type="florence2",
            eos_token_id=2,
            text_config=dict(
                d_model=16,
                encoder_attention_heads=2,
                decoder_attention_heads=2,
                encoder_ffn_dim=32,
                decoder_ffn_dim=32,
                encoder_layers=1,
                decoder_layers=1,
                vocab_size=64,
                max_position_embeddings=128,
                bos_token_id=1,
            ),
            vision_config=dict(
                model_type="davit",
                depths=[1] * 4,
                dim_embed=[8, 8, 16, 16],
                num_heads=[2] * 4,
                num_groups=[2] * 4,
                window_size=2,
                drop_path_rate=0,
                patch_size=[7, 3, 3, 3],
                patch_stride=[4, 2, 2, 2],
                patch_padding=[3, 1, 1, 1],
                patch_prenorm=[False, True, True, True],
            ),
        )
    if family == "paddleocr_vl":
        return dict(
            **{
                **text,
                "model_type": "paddleocr_vl",
                "head_dim": 8,
                "use_bias": False,
                "tie_word_embeddings": False,
            },
            rope_scaling=dict(type="default", mrope_section=[1, 1, 2]),
            image_token_id=4,
            vision_start_token_id=5,
            vision_end_token_id=6,
            vision_config=dict(
                model_type="paddleocr_vl",
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=2,
                image_size=8,
                spatial_merge_size=2,
            ),
        )
    if family == "lightonocr":
        return dict(
            model_type="mistral3",
            architectures=["LightOnOCRForConditionalGeneration"],
            image_token_id=4,
            text_config={**text, "model_type": "qwen3", "head_dim": 8},
            vision_config=dict(
                model_type="pixtral",
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                head_dim=8,
                patch_size=2,
                image_size=32,
            ),
        )
    if family == "deepseek_ocr":
        return dict(
            model_type="deepseekocr",
            eos_token_id=2,
            image_token_index=4,
            language_config={
                **text,
                "model_type": "deepseek_v2",
                "qk_nope_head_dim": 0,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "n_shared_experts": 1,
                "first_k_dense_replace": 0,
                "moe_intermediate_size": 8,
            },
            vision_config=dict(
                model_type="vision",
                layers=1,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=2,
                patch_size=4,
                params=dict(
                    base_size=64,
                    local_size=64,
                    position_image_size=4,
                    sam_config=dict(
                        img_size=64,
                        patch_size=16,
                        embed_dim=16,
                        depth=1,
                        num_heads=2,
                        window_size=2,
                        global_attn_indexes=[0],
                        out_chans=8,
                        final_out_chans=16,
                    ),
                ),
            ),
            projector_config=dict(input_dim=32, n_embed=16, projector_type="linear"),
        )
    raise ValueError(family)


def write_tiny_checkpoint(root, family, *, quantized=False, dtype=None):
    import mlx.core as mx
    import mlx.nn as nn
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

    from mlx_one.ocr.loading import make_config, make_model

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = tiny_config(family)
    model = make_model(make_config(payload, family), family)
    if quantized:
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0,
        )
        payload["quantization"] = dict(group_size=32, bits=4, mode="affine")
    if dtype is not None:
        model.apply(
            lambda value: value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value
        )
    model.save_weights(str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps(payload))
    images = {
        "got_ocr2": ["<imgpad>", "<img>", "</img>"],
        "paddleocr_vl": ["<|IMAGE_PLACEHOLDER|>", "<|IMAGE_START|>", "<|IMAGE_END|>"],
        "lightonocr": ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"],
    }.get(family, ["<image>", "<image_start>", "<image_end>"])
    vocab = {token: i for i, token in enumerate(["<unk>", "<s>", "</s>", "<pad>", *images])}
    vocab.update({f"t{i}": i for i in range(7, 64)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken(v, special=True) for v in list(vocab)[:7]])
    tok.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(dict(bos_token="<s>", eos_token="</s>", pad_token="<pad>"))
    )
    metadata = dict(
        size={"width": 64, "height": 64},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        min_pixels=64,
        max_pixels=256,
        resample=2,
    )
    if family == "lightonocr":
        metadata = {"image_processor": {**metadata, "size": {"longest_edge": 32}}}
    (root / "preprocessor_config.json").write_text(json.dumps(metadata))
    return root
