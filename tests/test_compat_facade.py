import sys
import types

import pytest

from mlx_one.compat.unsloth import FastLanguageModel


class FakeModel:
    layers = [object(), object()]


def test_fast_language_model_loading_subset_is_strict(monkeypatch) -> None:
    model = FakeModel()
    tokenizer = object()
    monkeypatch.setattr(
        "mlx_one.compat.unsloth.load_model",
        lambda *_args, **_kwargs: (model, tokenizer),
    )

    loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name="example/model",
        revision="a" * 40,
        max_seq_length=512,
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert model._mlx_one_revision == "a" * 40
    with pytest.raises(TypeError, match="unsupported"):
        FastLanguageModel.from_pretrained(model_name="example/model", unsupported=True)


def test_fast_language_model_peft_maps_to_upstream_lora(monkeypatch) -> None:
    calls = []
    utils = types.ModuleType("mlx_lm.tuner.utils")
    utils.linear_to_lora_layers = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner", types.ModuleType("mlx_lm.tuner"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", utils)
    model = FakeModel()

    configured = FastLanguageModel.get_peft_model(
        model,
        r=4,
        lora_alpha=8,
        target_modules=["q_proj"],
    )

    assert configured is model
    assert calls[0][1] == 2
    assert calls[0][2]["scale"] == 2
    assert model._mlx_one_peft_config["target_modules"] == ("q_proj",)
