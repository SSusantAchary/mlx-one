from pathlib import Path

import mlx_one.engine.public as public
from mlx_one import Engine
from mlx_one.text import GenerationChunk, GenerationResult, LoadedTextModel


class Tokenizer:
    def encode(self, value: str) -> list[int]:
        return list(range(len(value.split())))


def bundle() -> LoadedTextModel:
    return LoadedTextModel(
        model=object(),
        tokenizer=Tokenizer(),  # type: ignore[arg-type]
        path=Path("."),
        model_id="test/model",
        revision="revision",
        architecture="qwen3",
        context_length=4096,
        quantization={"bits": 4},
    )


def test_engine_context_manager_and_capabilities(monkeypatch) -> None:
    expected = GenerationResult("hi", "ok", 1, 1, "stop", "test/model")
    monkeypatch.setattr(public, "generate", lambda *args, **kwargs: expected)
    engine = Engine(bundle())
    with engine:
        assert engine.generate("hi", max_tokens=2) is expected
        assert engine.capabilities["architecture"] == "qwen3"
        assert engine.latest_result is expected
    try:
        engine.generate("again")
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("closed engine accepted generation")


def test_engine_stream_and_async_submission(monkeypatch) -> None:
    def chunks(*args, **kwargs):
        del args, kwargs
        yield GenerationChunk(1, "ok", 1)
        yield GenerationChunk(None, "", 1, "stop")

    monkeypatch.setattr(public, "stream_generate", chunks)
    engine = Engine(bundle())
    try:
        assert "".join(item.text for item in engine.stream("hi")) == "ok"
        result = engine.submit("hi", max_tokens=2).result(timeout=1)
        assert result.text == "ok"
        assert result.finish_reason == "stop"
    finally:
        engine.close()
