from types import SimpleNamespace

import pytest

import mlx_one.server.model_manager as model_manager


class NativeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self._vocabulary = vocabulary

    def get_vocab(self, *, with_added_tokens: bool) -> dict[str, int]:
        assert with_added_tokens
        return self._vocabulary


def loaded(name: str, vocabulary: dict[str, int]):
    return SimpleNamespace(
        model_id=name,
        architecture="qwen3",
        context_length=4096,
        parameter_count=10,
        quantization={},
        revision=None,
        tokenizer=SimpleNamespace(
            _tokenizer=NativeTokenizer(vocabulary),
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        ),
    )


def test_private_draft_requires_identical_token_ids(monkeypatch) -> None:
    bundles = {
        "primary": loaded("primary", {"a": 1, "b": 2}),
        "draft": loaded("draft", {"a": 1, "b": 2}),
        "bad": loaded("bad", {"a": 2, "b": 1}),
    }
    monkeypatch.setattr(model_manager, "load_text_model", lambda name, **_: bundles[name])
    manager = model_manager.ModelManager(alias="public")
    manager.load("primary")
    assert manager.load_draft("draft").model_id == "draft"
    assert manager.list_models()[0].id == "public"
    assert manager.draft_model().model_id == "draft"
    manager.unload()

    manager.load("primary")
    with pytest.raises(ValueError, match="identical token-ID"):
        manager.load_draft("bad")
