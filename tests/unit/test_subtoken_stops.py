"""Text stop strings may terminate inside or across tokenizer units."""

import pytest

from mlx_one.text import GenerationChunk, TextGenerationOptions
from mlx_one.text.generation import _stream_bundle


@pytest.mark.parametrize("pieces,stop,expected", [
    ([" Paris"], " Par", ""), (["prefix EN", "D ignored"], "END", "prefix "),
    (["x E", "N", "D suffix"], "END", "x "),
    (["hello E", "X"], "END", "hello EX"),
])
def test_stop_strings_do_not_leak_partial_matches(monkeypatch, pieces, stop, expected):
    closed = []

    def scripted(*args, **kwargs):
        try:
            for index, piece in enumerate(pieces):
                yield GenerationChunk(index, piece, index + 1)
            yield GenerationChunk(None, "", len(pieces), "length")
        finally:
            closed.append(True)

    monkeypatch.setattr("mlx_one.text.generation._stream_bundle_impl", scripted)
    chunks = list(_stream_bundle(None, "prompt", TextGenerationOptions(stop=(stop,)), None))
    assert "".join(chunk.text for chunk in chunks) == expected
    assert chunks[-1].finish_reason == ("stop" if stop != "END" or pieces[-1] != "X" else "length")
    assert closed == [True]
