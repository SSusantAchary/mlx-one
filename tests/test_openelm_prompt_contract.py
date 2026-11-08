"""OpenELM's required BOS must be present in generation and context accounting."""

from types import SimpleNamespace

import pytest

from mlx_one.engine.prompts import prepare_completion
from mlx_one.text.prompts import completion_token_ids


def bundle(tokens, *, bos=1, architecture="openelm"):
    return SimpleNamespace(
        architecture=architecture,
        context_length=8,
        tokenizer=SimpleNamespace(encode=lambda text: list(tokens), bos_token_id=bos),
    )


@pytest.mark.parametrize("architecture", ["openelm", "lfm2"])
def test_completion_has_one_bos_and_counts_it_against_context(architecture):
    model = bundle([4, 5], architecture=architecture)
    prepared = prepare_completion(model, "Hello world", 5, 8)
    assert prepared.token_ids == (1, 4, 5)
    with pytest.raises(ValueError, match="exceed"):
        prepare_completion(model, "Hello world", 6, 8)


def test_rendered_bos_is_not_duplicated_and_empty_encoding_has_a_start_token():
    assert completion_token_ids(bundle([1, 4]), "<s>Hello") == [1, 4]
    assert completion_token_ids(bundle([]), "text") == [1]


def test_incompatible_openelm_tokenizer_fails_closed():
    with pytest.raises(ValueError, match="BOS"):
        completion_token_ids(bundle([4], bos=None), "text")


def test_other_checkpoint_tokenization_is_preserved():
    assert completion_token_ids(bundle([4, 5], architecture="qwen3"), "text") == [4, 5]
