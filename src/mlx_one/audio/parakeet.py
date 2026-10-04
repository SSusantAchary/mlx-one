"""Native offline Parakeet TDT decoding, request-local predictor state and timestamps."""

from __future__ import annotations

import resource
import time
from pathlib import Path

from mlx_one.audio.moonshine import BUDGET, AudioResourceLimitError
from mlx_one.audio.schemas import ASRDecodeOptions, SpeechTranscript
from mlx_one.models.audio.parakeet.config import encoded_length
from mlx_one.models.audio.parakeet.processing import preprocess
from mlx_one.models.audio.whisper.processing import decode_audio


def timestamp_tokens(tokenizer, tokens, durations, *, blank, pad):
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream(skip_special_tokens=True)
    frame = 0
    spans = []
    for token, duration in zip(tokens, durations, strict=True):
        if token not in {blank, pad}:
            text = stream.step(tokenizer, token)
            if text is not None:
                start, end = frame * 0.08, (frame + duration) * 0.08
                if text in {"?", "'", "¡", "¿", "-", ":", ",", "%", "/", ".", "!"} and spans:
                    start = end = spans[-1]["end"]
                spans.append({"token": text, "start": start, "end": end})
        frame += duration
    return tuple(spans)


def greedy_decode(model, encoded, length, generation, *, limit=None, check=None):
    """No model-global cache and no returned silently truncated hypothesis."""
    import mlx.core as mx

    config = model.config
    blank = config.blank_token_id
    frame, previous = 0, blank
    state = decoder = None
    tokens, durations = [blank], [0]
    budget = limit if limit is not None else config.max_symbols_per_step * length
    try:
        for _ in range(budget):
            if frame >= length:
                break
            if check is not None:
                check()
            if decoder is None or previous != blank:
                decoder, state = model.decoder(previous, state)
            logits = model.joint(encoded[:, frame], decoder)
            token = int(mx.argmax(logits[:, : config.vocab_size], axis=-1).item())
            duration = int(mx.argmax(logits[:, config.vocab_size :], axis=-1).item())
            if token == blank and duration == 0:
                duration = 1
            tokens.append(token)
            durations.append(duration)
            frame += duration
            previous = token
            if token == generation.get("eos_token_id"):
                break
        else:
            if frame < length:
                raise AudioResourceLimitError(
                    f"TDT encoder not exhausted after {budget} emissions; refusing truncation"
                )
        return tuple(tokens), tuple(durations)
    finally:
        state = decoder = None


def transcribe_parakeet(
    bundle,
    audio,
    *,
    model_name,
    sample_rate,
    language,
    task,
    word_timestamps,
    options,
    cancelled,
    started,
    loaded_at,
) -> SpeechTranscript:
    if task != "transcribe" or word_timestamps:
        raise ValueError(
            "Parakeet supports offline transcription/token timestamps; word timestamps are WIP"
        )
    if options is not None and type(options) is not ASRDecodeOptions:
        raise ValueError("Parakeet accepts ASRDecodeOptions, not Whisper/Qwen options")
    if bundle.variant == "v2" and language not in {None, "en", "English", "english"}:
        raise ValueError("original Parakeet v2 is English-only")
    if bundle.variant == "v3" and language is not None:
        raise ValueError("Parakeet v3 does not implement explicit language conditioning")
    is_file = isinstance(audio, (str, Path))
    if sample_rate is not None and (isinstance(sample_rate, bool) or sample_rate != 16000):
        raise ValueError("Parakeet waveform sample_rate must be 16000; resample explicitly")
    if not is_file and sample_rate is None:
        raise ValueError("Parakeet waveform inputs require explicit sample_rate=16000")

    def check():
        if cancelled is not None and cancelled():
            raise RuntimeError("Parakeet transcription was cancelled")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise AudioResourceLimitError("24 GiB audio process-memory budget exceeded")

    check()
    import mlx.core as mx

    waveform = (
        decode_audio(audio, cancelled=cancelled) if is_file else mx.array(audio, dtype=mx.float32)
    )
    if waveform.ndim != 1 or not mx.all(mx.isfinite(waveform)).item():
        raise ValueError("Parakeet waveform must contain finite mono samples")
    duration = waveform.shape[0] / 16000
    tokens, durations = (), ()
    processed_at = time.perf_counter()
    if waveform.shape[0]:
        frames = waveform.shape[0] // 160
        length = encoded_length(frames + 1)
        working = 5 * 4 * bundle.model.config.num_attention_heads * length**2
        if working + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise AudioResourceLimitError("Parakeet dense encoder would exceed 24 GiB budget")
        features, valid_frames = preprocess(waveform, bundle.processor)
        mx.eval(features)
        processed_at = time.perf_counter()
        check()
        encoded, lengths = bundle.model.encode(features, mx.array([valid_frames]), check=check)
        mx.eval(encoded, lengths)
        tokens, durations = greedy_decode(
            bundle.model,
            encoded,
            int(lengths[0].item()),
            bundle.generation_config,
            limit=options.max_tokens if options else None,
            check=check,
        )
        check()
    completed = time.perf_counter()
    spans = timestamp_tokens(
        bundle.tokenizer,
        tokens,
        durations,
        blank=bundle.model.config.blank_token_id,
        pad=bundle.model.config.pad_token_id,
    )
    return SpeechTranscript(
        bundle.tokenizer.decode(list(tokens), skip_special_tokens=True),
        duration,
        model_name,
        bundle.revision,
        16000,
        tokens=tokens,
        language="en" if bundle.variant == "v2" and language is not None else None,
        timestamps=spans,
        timings={
            "load_seconds": loaded_at - started,
            "preprocess_seconds": processed_at - loaded_at,
            "decode_seconds": completed - processed_at,
            "wall_seconds": completed - started,
        },
        provenance={
            "model_type": "parakeet_tdt",
            "variant": bundle.variant,
            "checkpoint_path": str(bundle.path),
            "checkpoint_revision": bundle.revision,
            "weight_dtype": str(bundle.model.encoder_projector.weight.dtype),
            "decoding": "greedy-tdt",
            "streaming": False,
            "timestamp_kind": "token",
            "timestamp_tick_seconds": 0.08,
            "qualification": "WIP",
        },
    )
