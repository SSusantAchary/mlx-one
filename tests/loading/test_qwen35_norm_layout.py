"""Publisher MLX folded scales must map to the native HF offset norm."""

import numpy as np
import pytest
from test_native_qwen_multimodal_architectures import qwen3_5_config

from mlx_one.models.vision_language.qwen3_5.weights import sanitize_weights


@pytest.mark.parametrize("layout,folded", [((1, 4), False), ((4, 1), True), ((4,), False)])
def test_norm_offset_conversion_is_bound_to_source_convolution_layout(layout, folded):
    config = qwen3_5_config()
    prefix = "model.language_model."
    names = ["layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight",
             "layers.3.self_attn.q_norm.weight", "layers.3.self_attn.k_norm.weight", "norm.weight"]
    source = {prefix + name: np.full((16,), 0.25, dtype=np.float32) for name in names}
    gated = prefix + "layers.0.linear_attn.norm.weight"
    source[gated] = np.full((4,), 0.75, dtype=np.float32)
    source[prefix + "layers.0.linear_attn.conv1d.weight"] = np.zeros((24, *layout))
    result = sanitize_weights(source, config)
    for name in names:
        np.testing.assert_array_equal(result["language_model." + name],
                                      np.full((16,), -0.75 if folded else 0.25))
    np.testing.assert_array_equal(result[gated.removeprefix("model.")], source[gated])
    # Already sanitized native tensors are not offset again.
    twice = sanitize_weights(result, config)
    for name, value in result.items():
        np.testing.assert_array_equal(twice[name], value)
    assert source[prefix + "norm.weight"][0] == 0.25


def test_mixed_source_layouts_are_rejected_instead_of_guessing_norm_convention():
    with pytest.raises(ValueError, match="mixed HF and MLX"):
        sanitize_weights({
            "model.language_model.layers.0.linear_attn.conv1d.weight": np.zeros((24, 1, 4)),
            "model.language_model.layers.1.linear_attn.conv1d.weight": np.zeros((24, 4, 1)),
        }, qwen3_5_config())
