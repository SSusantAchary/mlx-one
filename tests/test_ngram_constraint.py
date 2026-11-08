"""Publisher no-repeat decoding constrains the next token without erasing history."""

import pytest

from mlx_one.text.sampling import banned_ngram_tokens


def test_ngram_suffix_only_blocks_tokens_that_repeat_a_complete_ngram():
    assert banned_ngram_tokens([1, 2, 1, 2], 2) == {1}
    assert banned_ngram_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert banned_ngram_tokens([1, 2], 4) == set()
    assert banned_ngram_tokens([1, 2, 1], 1) == {1, 2}
    assert banned_ngram_tokens([], 1) == set()


def test_ngram_constraint_matches_publisher_twenty_token_repetition_rule():
    history = list(range(20)) + list(range(19))
    assert banned_ngram_tokens(history, 20) == {19}
    for invalid in [True, 0, -1, 2.5]:
        with pytest.raises(ValueError):
            banned_ngram_tokens(history, invalid)
