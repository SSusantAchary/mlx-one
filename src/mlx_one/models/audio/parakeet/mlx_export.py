"""Explicit MLX community Parakeet export mapping to the existing native model."""

from __future__ import annotations

from mlx_one.models.audio.parakeet.config import ParakeetConfig


def config_from_mlx(data):
    if (
        data.get("model_defaults", {}).get("tdt_durations") != [0, 1, 2, 3, 4]
        or data.get("encoder", {}).get("subsampling_factor") != 8
        or data.get("encoder", {}).get("self_attention_model") != "rel_pos"
        or data.get("sample_rate") != 16000
        or data.get("quantization")
        or data.get("quantization_config")
    ):
        raise ValueError("unsupported MLX Parakeet export configuration")
    enc, decoder = data["encoder"], data["decoder"]
    if (
        enc.get("use_bias", False)
        or enc.get("causal_downsampling", False)
        or enc.get("reduction") is not None
        or enc.get("conv_norm_type") != "batch_norm"
    ):
        raise ValueError("unsupported MLX Parakeet encoder variant")
    vocabulary = data["joint"]["vocabulary"]
    if decoder["vocab_size"] != len(vocabulary):
        raise ValueError("Parakeet vocabulary length disagrees with decoder")
    return ParakeetConfig(
        vocab_size=len(vocabulary) + 1,
        blank_token_id=len(vocabulary),
        pad_token_id=vocabulary.index("<pad>") if "<pad>" in vocabulary else len(vocabulary),
        decoder_hidden_size=decoder["prednet"]["pred_hidden"],
        num_decoder_layers=decoder["prednet"]["pred_rnn_layers"],
        hidden_size=enc["d_model"],
        intermediate_size=enc["d_model"] * enc["ff_expansion_factor"],
        num_hidden_layers=enc["n_layers"],
        num_attention_heads=enc["n_heads"],
        num_mel_bins=enc["feat_in"],
        conv_kernel_size=enc["conv_kernel_size"],
        subsampling_conv_channels=enc["subsampling_conv_channels"],
        scale_input=enc.get("xscaling", False),
        extra=dict(data),
    )


def map_weights(weights, config):
    import mlx.core as mx

    from mlx_one.models.audio.parakeet.weights import weight_contract

    result = {}
    for name, value in weights.items():
        target = name.replace("encoder.pre_encode.conv.", "encoder.subsampling.layers.")
        target = target.replace("encoder.pre_encode.out.", "encoder.subsampling.linear.")
        target = target.replace(".conv.batch_norm.", ".conv.norm.")
        for a, b in [
            ("linear_q", "q_proj"),
            ("linear_k", "k_proj"),
            ("linear_v", "v_proj"),
            ("linear_out", "o_proj"),
            ("linear_pos", "relative_k_proj"),
            ("pos_bias_u", "bias_u"),
            ("pos_bias_v", "bias_v"),
        ]:
            target = target.replace(".self_attn." + a, ".self_attn." + b)
        target = target.replace("decoder.prediction.embed.", "decoder.embedding.")
        target = target.replace("joint.enc.", "encoder_projector.")
        target = target.replace("joint.pred.", "decoder.decoder_projector.")
        target = target.replace("joint.joint_net.2.", "joint.head.")
        prefix = "decoder.prediction.dec_rnn.lstm."
        if name.startswith(prefix):
            layer, parameter = name[len(prefix) :].split(".")
            suffix = {"Wx": "weight_ih", "Wh": "weight_hh", "bias": "bias_ih"}[parameter]
            target = f"decoder.lstm.{suffix}_l{layer}"
            if parameter == "bias":
                result[f"decoder.lstm.bias_hh_l{layer}"] = mx.zeros_like(value)
        if target in result:
            raise ValueError(f"duplicate mapped Parakeet tensor: {target}")
        result[target] = value
    for i in range(config.num_hidden_layers):
        result[f"encoder.layers.{i}.conv.norm.num_batches_tracked"] = mx.array(0, dtype=mx.int64)
    weight_contract(config).validate(result)
    return result


class VocabularyTokenizer:
    """SentencePiece pieces from the export config, including its exact special IDs."""

    def __init__(self, pieces):
        from tokenizers import Tokenizer, decoders, models

        self.native = Tokenizer(models.WordLevel({p: i for i, p in enumerate(pieces)}))
        self.native.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        pieces = [self.pieces[i] for i in ids if 0 <= i < len(self.pieces)]
        if skip_special_tokens:
            pieces = [p for p in pieces if not p.startswith("<")]
        return "".join(pieces).replace("▁", " ").lstrip()
