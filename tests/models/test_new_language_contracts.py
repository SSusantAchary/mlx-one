"""Backend-free exact dimension/key contracts for the two new model families."""

import pytest

from mlx_one.core.config import ConfigError
from mlx_one.core.registry import get_registration
from mlx_one.models.language.gemma3.config import Gemma3TextConfig
from mlx_one.models.language.phi3.config import Phi3Config


def tiny_config(family, **updates):
    data = dict(
        model_type=family,
        hidden_size=16,
        num_hidden_layers=3,
        intermediate_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    if family == "gemma3_text":
        data.update(head_dim=4, sliding_window=4, sliding_window_pattern=3, query_pre_attn_scalar=4)
    data.update(updates)
    return get_registration(family).config_class().from_dict(data)


@pytest.mark.parametrize("family,cls", [("phi3", Phi3Config), ("gemma3_text", Gemma3TextConfig)])
def test_family_registration_has_native_loader_and_strict_weight_contract(family, cls):
    entry = get_registration(family)
    config = tiny_config(family)
    assert isinstance(config, cls)
    assert entry.supports("text-generation") and not entry.supports("image")
    contract = entry.weight_contract()(config)
    assert contract.expected["model.embed_tokens.weight"] == (64, 16)
    assert "lm_head.weight" not in contract.expected
    if family == "phi3":
        assert contract.expected["model.layers.0.self_attn.qkv_proj.weight"] == (32, 16)
        assert contract.expected["model.layers.0.mlp.gate_up_proj.weight"] == (48, 16)
    else:
        assert config.layer_types == ("sliding_attention", "sliding_attention", "full_attention")
        assert contract.expected["model.layers.0.pre_feedforward_layernorm.weight"] == (16,)


@pytest.mark.parametrize(
    "update",
    [
        dict(head_dim=3),
        dict(sliding_window=0),
        dict(layer_types=["full_attention"]),
        dict(attention_bias=True),
        dict(use_bidirectional_attention=True),
    ],
)
def test_gemma_rejects_unsupported_mathematics(update):
    with pytest.raises(ConfigError):
        tiny_config("gemma3_text", **update)


@pytest.mark.parametrize(
    "update",
    [
        dict(partial_rotary_factor=0.3),
        dict(hidden_size=15),
        dict(hidden_act="gelu"),
        dict(sliding_window=0),
        dict(hidden_size=16.0),
        dict(rope_scaling={"type": "longrope", "long_factor": [1]}),
    ],
)
def test_phi3_rejects_incompatible_packed_or_rotary_layouts(update):
    with pytest.raises(ConfigError):
        tiny_config("phi3", **update)


def test_sliding_cache_cannot_be_replaced_by_dense_quantized_storage():
    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.core.sliding_cache import SlidingKVCache

    cache = SlidingKVCache(4)
    assert configure_kv_caches((cache,), "bf16", "bf16")[0] is cache
    assert cache.storage_dtype == "bfloat16"
    with pytest.raises(ValueError, match="sliding KV"):
        configure_kv_caches((cache,), "q4_0", "q4_0")


def test_phi3_maps_exact_checkpoint_window_instead_of_rejecting_it():
    assert tiny_config("phi3", sliding_window=2047).sliding_window == 2047


def test_optional_template_defaults_do_not_create_tool_call_membership():
    from mlx_one.text.chat import ChatTemplate

    template = (
        "{% for message in messages %}"
        "{% if 'tool_calls' in message %}TOOL{% else %}{{ message.content }}{% endif %}"
        "{% if message.tool_calls %}CALL{% endif %}{% endfor %}"
    )
    chat = ChatTemplate("llama", template, None)
    assert chat.render([{"role": "user", "content": "hello"}]) == "hello"
    assert chat.render([{"role": "assistant", "content": "", "tool_calls": [1]}]) == "TOOLCALL"


@pytest.mark.parametrize(
    "family,window,expected",
    [
        ("phi3", 2047, False),
        ("phi3", None, True),
        ("gemma3_text", 512, False),
    ],
)
def test_public_capabilities_use_the_loaded_model_config(family, window, expected):
    from pathlib import Path
    from types import SimpleNamespace

    from mlx_one import Engine
    from mlx_one.text import LoadedTextModel

    engine = object.__new__(Engine)
    engine._bundle = LoadedTextModel(
        SimpleNamespace(config=SimpleNamespace(sliding_window=window)),
        None,
        Path("."),
        "test/native",
        "revision",
        architecture=family,
    )
    engine._closed = False
    engine._cache_manager = None
    engine._inference_mode = "reference_dense"
    engine.runtime_stats = lambda: {}
    assert engine.capabilities()["cache"]["quantized_kv"] is expected


@pytest.mark.parametrize("family", ["phi3", "gemma3_text"])
def test_new_families_do_not_invent_missing_chat_templates(family):
    from mlx_one.text.chat import ChatTemplate

    with pytest.raises(ValueError, match="chat template"):
        ChatTemplate(family, None, None).render([{"role": "user", "content": "hi"}])
