import json
from pathlib import Path

import pytest

from mlx_one.core.registry import get_registration
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.loading import _quantization_config, _validate_quantized_tensors
from mlx_one.text.tokenizers import HFTokenizerAdapter


class TinyTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    special_tokens = {"bos_token": "<s>", "eos_token": "</s>"}


@pytest.mark.parametrize(
    "model_type",
    ["gpt2", "lfm2", "lfm2_moe", "openelm", "qwen2", "qwen2_moe", "qwen3", "qwen3_5"],
)
def test_native_serving_families_have_loading_hooks(model_type: str) -> None:
    registration = get_registration(model_type)
    assert callable(registration.sanitizer())
    assert callable(registration.weight_contract())
    assert registration.capabilities >= {"generate", "sample", "stream"}


def test_chat_template_fallbacks_and_restricted_template(tmp_path: Path) -> None:
    qwen = ChatTemplate("qwen3", None, TinyTokenizer())
    assert qwen.render([{"role": "user", "content": "Hello"}]).endswith(
        "<|im_start|>assistant\n"
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages[0]['content'] }} :: assistant"}),
        encoding="utf-8",
    )
    template = ChatTemplate.from_directory(tmp_path, "gpt2", TinyTokenizer())
    assert template.render([{"role": "user", "content": "Hello"}]) == "Hello :: assistant"


def test_standalone_chat_template_preserves_required_bos_token(tmp_path: Path) -> None:
    (tmp_path / "chat_template.jinja").write_text(
        "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}",
        encoding="utf-8",
    )

    template = ChatTemplate.from_directory(tmp_path, "lfm2", TinyTokenizer())

    assert template.render([{"role": "user", "content": "Hello"}]) == "<s>Hello"


def test_hf_tokenizer_adapter_exposes_special_tokens(tmp_path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "<s>": 1, "</s>": 2, "hello": 3}, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "[UNK]"}),
        encoding="utf-8",
    )
    adapter = HFTokenizerAdapter.from_directory(tmp_path)
    assert adapter.encode("hello") == [3]
    assert adapter.decode([3]) == "hello"
    assert adapter.bos_token_id == 1
    assert adapter.eos_token_id == 2
    assert adapter.special_tokens["unk_token"] == "[UNK]"


def test_quantization_metadata_is_strict() -> None:
    assert _quantization_config({"quantization": {"bits": 4, "group_size": 64}}) == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    with pytest.raises(ValueError, match="only MLX 4-bit"):
        _quantization_config({"quantization": {"bits": 8}})
    with pytest.raises(ValueError, match="only MLX 4-bit"):
        _quantization_config({"quantization_config": {"quant_method": "bitsandbytes"}})
    with pytest.raises(ValueError, match="unsupported quantization method"):
        _quantization_config(
            {"quantization_config": {"bits": 4, "quant_method": "bitsandbytes"}}
        )


def test_mixed_quantized_tensor_tree_is_strict() -> None:
    class Tensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class TinyModel:
        def parameters(self):
            return {
                "quantized": {
                    "weight": Tensor((2, 1)),
                    "scales": Tensor((2, 1)),
                    "biases": Tensor((2, 1)),
                },
                "plain": {"weight": Tensor((2, 2))},
            }

    tensors = {
        "quantized.weight": Tensor((2, 1)),
        "quantized.scales": Tensor((2, 1)),
        "quantized.biases": Tensor((2, 1)),
        "plain.weight": Tensor((2, 2)),
    }
    _validate_quantized_tensors(TinyModel(), tensors)
    with pytest.raises(ValueError, match="shape_mismatch"):
        _validate_quantized_tensors(TinyModel(), {**tensors, "plain.weight": Tensor((3, 2))})
