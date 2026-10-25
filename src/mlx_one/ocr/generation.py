"""OCR page preparation over MLX-One's existing cached text-generation machinery."""

from __future__ import annotations

import time
from dataclasses import replace

import mlx.core as mx

from mlx_one.core.outputs import ModelOutput
from mlx_one.engine.cache import CacheBundle
from mlx_one.models.ocr._layers import KVCache
from mlx_one.ocr.postprocessing import parse_output
from mlx_one.ocr.processing import prepare_ocr
from mlx_one.ocr.schemas import OCRPage
from mlx_one.text.loading import LoadedTextModel
from mlx_one.text.schemas import TextGenerationOptions


class _RawTokenizer:
    """Florence location tokens carry predictions and must survive generation decoding."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def decode(self, ids, **_):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


class OCRForward:
    """Request-local visual embeddings / encoder state; one shared native decoder."""

    def __init__(self, bundle, prepared):
        self.bundle, self.prepared = bundle, prepared
        self.config = getattr(
            bundle.config, "text_config", getattr(bundle.config, "decoder", bundle.config)
        )
        self.encoder_state = None
        self.embeddings = None
        family, model, args = bundle.architecture, bundle.model, prepared.kwargs
        ids = mx.array([prepared.ids])
        if family == "trocr":
            self.encoder_state = model.encode(args["pixel_values"])
        elif family == "florence2":
            features = model.get_input_embeddings(args["encoder_input_ids"], args["pixel_values"])
            lm = model.language_model.model
            lm.encoder.embed_tokens = lm.shared
            self.encoder_state = lm.encoder(inputs_embeds=features.inputs_embeds)
        else:
            self.embeddings = model.get_input_embeddings(ids, **args).inputs_embeds
        mx.eval(self.encoder_state if self.encoder_state is not None else self.embeddings)

    def make_cache(self):
        if self.bundle.architecture in {"trocr", "florence2"}:
            return tuple(self.bundle.model.make_cache())
        return tuple(KVCache() for _ in range(self.config.num_hidden_layers))

    def __call__(self, ids, *, cache, logits_mode=None, **_):
        family, model = self.bundle.architecture, self.bundle.model
        initial = cache[0].offset == 0
        if family == "trocr":
            output = model.decode(ids, self.encoder_state, cache)
        elif family == "florence2":
            output = model.language_model(
                decoder_input_ids=ids, encoder_outputs=self.encoder_state, cache=cache
            )
        elif family == "lightonocr":
            output = model.language_model(
                None if initial else ids,
                cache=cache,
                input_embeddings=self.embeddings if initial else None,
            )
        else:
            output = model.language_model(
                ids, cache=cache, inputs_embeds=self.embeddings if initial else None
            )
        logits = output.logits
        return ModelOutput(logits[:, -1:, :], logits[:, -1:, :], cache=cache)

    def close(self):
        self.embeddings = self.encoder_state = self.prepared = None
        if self.bundle.architecture == "paddleocr_vl":
            self.bundle.model.language_model._position_ids = None
            self.bundle.model.language_model._rope_deltas = None


def stream_page(bundle, image, request, *, page=0, page_metadata=None, cancelled=lambda: False):
    """Yield native chunks, then return an OCRPage. Always release generation state."""
    from mlx_one.text.generation import _stream_bundle

    started = time.perf_counter()
    if cancelled():
        raise TimeoutError("OCR page was cancelled or timed out")
    prepared = prepare_ocr(bundle, image, request)
    prepared_seconds = time.perf_counter() - started
    forward, state, stream = None, None, None
    try:
        encoded = time.perf_counter()
        forward = OCRForward(bundle, prepared)
        encoder_seconds = time.perf_counter() - encoded
        state = CacheBundle(forward.make_cache())
        tokenizer = (
            _RawTokenizer(bundle.tokenizer)
            if bundle.architecture == "florence2"
            else (bundle.tokenizer)
        )
        text = LoadedTextModel(
            forward,
            tokenizer,
            bundle.path,
            bundle.model_id,
            bundle.revision,
            bundle.architecture,
            bundle.context_length,
            eos_token_ids=bundle.eos_token_ids,
        )
        forced = (
            (bundle.config.text_config.bos_token_id,)
            if (bundle.architecture == "florence2")
            else ()
        )
        stages = {"text_prefill_seconds": 0.0, "decode_forward_seconds": 0.0}

        def observe(event):
            if event.get("kind") == "forward":
                name = (
                    "text_prefill_seconds"
                    if event["phase"] == "prefill"
                    else "decode_forward_seconds"
                )
                stages[name] += event["duration_ms"] / 1000

        generation_started = time.perf_counter()
        # ByteFallback can temporarily replace a whole byte run (including an
        # already decoded newline) when a later UTF-8 code point is incomplete.
        # Public OCR streaming emits complete pages, so decode Paddle's token
        # sequence once at page completion instead of committing unstable text.
        buffered_decode = bundle.architecture == "paddleocr_vl"
        stream = _stream_bundle(
            text,
            "",
            TextGenerationOptions(max_tokens=request.max_tokens, temperature=0),
            cancelled,
            cache=state.entries,
            _prepared_input_ids=prepared.ids,
            _forced_token_ids=forced,
            on_execution=observe,
            _token_events_only=buffered_decode,
        )
        chunks = []
        first = None
        for chunk in stream:
            if first is None:
                first = time.perf_counter() - started
            if buffered_decode and chunk.finish_reason is not None:
                token_ids = [c.token_id for c in chunks if c.token_id is not None]
                chunk = replace(chunk, text=tokenizer.decode(token_ids))
            chunks.append(chunk)
            yield chunk
        if cancelled():
            raise TimeoutError("OCR page was cancelled or timed out")
        generation_seconds = time.perf_counter() - generation_started
        raw = "".join(c.text for c in chunks)
        text, markdown, elements, tables, formulas = parse_output(
            bundle, request, raw, prepared, page
        )
        return OCRPage(
            page,
            image.size,
            image.size,
            text,
            markdown,
            elements,
            tables,
            formulas,
            {
                "preprocessing_seconds": prepared_seconds,
                "encoder_seconds": encoder_seconds,
                "generation_seconds": generation_seconds,
                **stages,
                "total_seconds": time.perf_counter() - started,
                "first_token_seconds": first,
            },
            {
                **(page_metadata or {}),
                **prepared.metadata,
                "model": bundle.model_id,
                "revision": bundle.revision,
                "architecture": bundle.architecture,
                "output_tokens": chunks[-1].generated_tokens if chunks else 0,
                "finish_reason": chunks[-1].finish_reason if chunks else None,
                "truncated": bool(chunks and chunks[-1].finish_reason == "length"),
                "geometry_predicted": bool(elements),
                "confidence_available": False,
            },
        )
    finally:
        if stream is not None:
            stream.close()
        if state is not None:
            state.release()
        if forward is not None:
            forward.close()
        elif bundle.architecture == "paddleocr_vl":
            bundle.model.language_model._position_ids = None
            bundle.model.language_model._rope_deltas = None
