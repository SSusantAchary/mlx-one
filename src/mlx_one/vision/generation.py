"""Cached native generation for Qwen vision-language bundles."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from mlx_one.text.sampling import select_token
from mlx_one.text.schemas import GenerationChunk, TextGenerationOptions
from mlx_one.vision.loading import LoadedVLM


def stream_vlm(
    bundle: LoadedVLM,
    prompt: str,
    images: Sequence[object],
    *,
    options: TextGenerationOptions | None = None,
) -> Iterator[GenerationChunk]:
    """Stream a completion after one multimodal prefill and cached token decoding."""
    import mlx.core as mx

    settings = options or TextGenerationOptions()
    prepared = bundle.processor.prepare(prompt, tuple(images))
    cache = bundle.model.make_cache()
    output = bundle.model(
        prepared.input_ids,
        pixel_values=prepared.pixel_values,
        image_grid_thw=prepared.image_grid_thw,
        mm_token_type_ids=prepared.mm_token_type_ids,
        cache=cache,
    )
    rng = random.Random(settings.seed)
    generated = []
    decoded = ""
    eos = bundle.tokenizer.eos_token_id
    for index in range(settings.max_tokens):
        token = select_token(output.logits[0, -1], settings, rng)
        if eos is not None and token == eos:
            yield GenerationChunk(None, "", index, "stop")
            return
        generated.append(token)
        text = bundle.tokenizer.decode(generated)
        piece = text[len(decoded) :] if text.startswith(decoded) else text
        decoded = text
        yield GenerationChunk(token, piece, index + 1)
        output = bundle.model(mx.array([[token]], dtype=mx.int32), cache=cache)
    yield GenerationChunk(None, "", len(generated), "length")
