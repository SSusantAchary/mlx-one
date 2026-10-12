"""Exact HF tensor contracts; convolution layout changes only, no dtype changes."""

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.audio.parakeet.config import ParakeetConfig


def weight_contract(config: ParakeetConfig, *, native: bool = True) -> WeightContract:
    d, f, c, h = (
        config.hidden_size,
        config.intermediate_size,
        config.subsampling_conv_channels,
        config.decoder_hidden_size,
    )
    expected = {}

    def linear(prefix: str, out: int, inp: int, bias: bool) -> None:
        expected[prefix + ".weight"] = (out, inp)
        if bias:
            expected[prefix + ".bias"] = (out,)

    def conv(prefix: str, out: int, inp: int, kernel: int, bias: bool, dimensions: int) -> None:
        shape = (out, *((kernel,) * dimensions), inp)
        if not native:
            shape = (out, inp, *((kernel,) * dimensions))
        expected[prefix + ".weight"] = shape
        if bias:
            expected[prefix + ".bias"] = (out,)

    for index, inp, kernel in ((0, 1, 3), (2, 1, 3), (3, c, 1), (5, 1, 3), (6, c, 1)):
        conv(f"encoder.subsampling.layers.{index}", c, inp, kernel, True, 2)
    linear("encoder.subsampling.linear", d, c * (config.num_mel_bins // 8), True)
    for index in range(config.num_hidden_layers):
        prefix = f"encoder.layers.{index}"
        for ff in ("feed_forward1", "feed_forward2"):
            linear(f"{prefix}.{ff}.linear1", f, d, config.attention_bias)
            linear(f"{prefix}.{ff}.linear2", d, f, config.attention_bias)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"):
            linear(
                f"{prefix}.self_attn.{proj}",
                d,
                d,
                config.attention_bias and proj != "relative_k_proj",
            )
        for bias in ("bias_u", "bias_v"):
            expected[f"{prefix}.self_attn.{bias}"] = (
                config.num_attention_heads,
                d // config.num_attention_heads,
            )
        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            for key in ("weight", "bias"):
                expected[f"{prefix}.{norm}.{key}"] = (d,)
        conv(f"{prefix}.conv.pointwise_conv1", 2 * d, d, 1, config.convolution_bias, 1)
        conv(
            f"{prefix}.conv.depthwise_conv",
            d,
            1,
            config.conv_kernel_size,
            config.convolution_bias,
            1,
        )
        conv(f"{prefix}.conv.pointwise_conv2", d, d, 1, config.convolution_bias, 1)
        for key in ("weight", "bias", "running_mean", "running_var"):
            expected[f"{prefix}.conv.norm.{key}"] = (d,)
        expected[f"{prefix}.conv.norm.num_batches_tracked"] = ()
    expected["decoder.embedding.weight"] = (config.vocab_size, h)
    for index in range(config.num_decoder_layers):
        for side in ("ih", "hh"):
            expected[f"decoder.lstm.weight_{side}_l{index}"] = (4 * h, h)
            expected[f"decoder.lstm.bias_{side}_l{index}"] = (4 * h,)
    linear("decoder.decoder_projector", h, h, True)
    linear("encoder_projector", h, d, True)
    linear("joint.head", config.vocab_size + len(config.durations), h, True)
    return WeightContract(expected)


def sanitize_weights(weights: dict[str, Any], config: ParakeetConfig) -> dict[str, Any]:
    weight_contract(config, native=False).validate(weights)
    result = {}
    for name, value in weights.items():
        if name.endswith(".weight") and ".subsampling.layers." in name:
            value = value.transpose(0, 2, 3, 1)
        elif name.endswith(".weight") and any(
            f".conv.{part}." in name
            for part in ("pointwise_conv1", "pointwise_conv2", "depthwise_conv")
        ):
            value = value.transpose(0, 2, 1)
        result[name] = value
    return result
