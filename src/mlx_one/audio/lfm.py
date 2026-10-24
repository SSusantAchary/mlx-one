"""Native LFM Audio ASR, sequential TTS and interleaved speech responses."""

from __future__ import annotations

import time

from mlx_one.audio.schemas import SpeechGenerationResult, SpeechStreamEvent, SpeechTranscript
from mlx_one.models.audio.lfm2_audio.loading import LoadedLFMAudio

VOICES = {
    "us_male": "US male",
    "us_female": "US female",
    "uk_male": "UK male",
    "uk_female": "UK female",
}


def _prompt(bundle, task, *, text="", waveform=None, voice="us_male", timings=None, cancelled=None):
    import mlx.core as mx

    from mlx_one.models.audio.lfm2_audio.dsp import preprocess

    if task == "synthesize":
        if voice not in VOICES:
            raise ValueError("LFM voice must be one of: " + ", ".join(VOICES))
        system = f"Perform TTS. Use the {VOICES[voice]} voice."
    elif task == "transcribe":
        system = "Perform ASR."
    else:
        system = "Respond with interleaved text and audio."
    prefix = f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"

    def encode(value):
        return bundle.tokenizer.encode(value, add_special_tokens=False).ids

    pieces = [bundle.model.lfm.embed_tokens(mx.array([encode(prefix)]))]
    if waveform is not None:
        from mlx_one.audio.qwen import check_resources

        frontend_started = time.perf_counter()
        features = preprocess(mx.array(waveform), bundle.model.config.preprocessor)
        mx.eval(features)
        encoder_started = time.perf_counter()
        audio, _ = bundle.model.encode(features, check=lambda: check_resources(cancelled))
        mx.eval(audio)
        if timings is not None:
            timings["frontend_seconds"] = encoder_started - frontend_started
            timings["encoder_seconds"] = time.perf_counter() - encoder_started
        pieces.append(audio.astype(pieces[0].dtype))
    else:
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise ValueError("speech text must contain 1 to 16000 characters")
        pieces.append(bundle.model.lfm.embed_tokens(mx.array([encode(text)])))
    pieces.append(
        bundle.model.lfm.embed_tokens(mx.array([encode("<|im_end|>\n<|im_start|>assistant\n")]))
    )
    embeddings = mx.concatenate(pieces, axis=1)
    if embeddings.shape[1] > min(8192, bundle.model.config.lfm.max_position_embeddings):
        raise ValueError("LFM audio prompt exceeds its bounded context")
    mx.eval(embeddings)
    return embeddings


def stream_lfm(
    bundle,
    *,
    task="synthesize",
    text="",
    audio=None,
    sample_rate=None,
    voice="us_male",
    max_tokens=1024,
    cancelled=None,
    chunk_frames=12,
    audio_temperature=0.0,
    audio_top_k=0,
):
    """Causal model state; bounded codec prefix recomputation, not a codec KV cache.

    Audio chunks contain stable PCM only. The overlapping ISTFT tail is withheld
    until subsequent frames or final flush. No EOS code is sent to the codec.
    """
    import mlx.core as mx
    import numpy as np

    from mlx_one.audio.input import prepare_audio
    from mlx_one.audio.qwen import check_resources

    if not isinstance(bundle, LoadedLFMAudio):
        raise ValueError("speech generation requires a native LFM Audio checkpoint")
    if task not in {"synthesize", "speech-to-speech", "transcribe"}:
        raise ValueError("unsupported LFM audio task")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be between 1 and 4096")
    if type(chunk_frames) is not int or not 1 <= chunk_frames <= 32:
        raise ValueError("codec chunk_frames must be between 1 and 32")
    if not np.isfinite(audio_temperature) or audio_temperature < 0:
        raise ValueError("audio_temperature must be finite and non-negative")
    if type(audio_top_k) is not int or audio_top_k < 0:
        raise ValueError("audio_top_k must be a non-negative integer")
    started = time.perf_counter()
    check_resources(cancelled)
    waveform = (
        prepare_audio(audio, sample_rate=sample_rate, max_seconds=180, cancelled=cancelled)
        if audio is not None
        else None
    )
    if task != "synthesize" and waveform is None:
        raise ValueError("this LFM task requires audio input")
    timings = {"input_decode_seconds": time.perf_counter() - started}
    embeddings = _prompt(
        bundle,
        task,
        text=text,
        waveform=waveform,
        voice=voice,
        timings=timings,
        cancelled=cancelled,
    )
    processed = time.perf_counter()
    sequence, emitted, generated = 0, 0, 0
    frames = []
    span_base = 0
    timings["preprocess_seconds"] = processed - started
    text_tokens = []
    last_text = ""
    iterator = bundle.model.generate(
        embeddings,
        mode="interleaved" if task == "speech-to-speech" else "sequential",
        max_tokens=max_tokens,
        cancelled=cancelled,
        audio_temperature=audio_temperature,
        audio_top_k=audio_top_k,
    )

    def decode(final=False):
        nonlocal emitted, sequence
        if not frames:
            return None
        codes = mx.stack(frames, axis=-1)
        codec_started = time.perf_counter()
        pcm = bundle.codec(codes).reshape(-1)
        mx.eval(pcm)
        timings["codec_decode_seconds"] = timings.get("codec_decode_seconds", 0.0) + (
            time.perf_counter() - codec_started
        )
        check_resources(cancelled)
        stable = pcm.size if final else max(0, pcm.size - bundle.codec.n_fft)
        if stable <= emitted:
            return None
        data = np.array(pcm[emitted:stable], dtype=np.float32)
        if not data.size or not np.isfinite(data).all():
            raise ValueError("LFM codec produced an invalid waveform")
        timings.setdefault("time_to_first_audio_seconds", time.perf_counter() - started)
        event = SpeechStreamEvent(
            "audio_chunk",
            sequence,
            waveform=data,
            sample_offset=span_base + emitted,
            timings=dict(timings),
        )
        emitted, sequence = stable, sequence + 1
        timings["audio_chunk_count"] = timings.get("audio_chunk_count", 0) + 1
        return event

    try:
        for kind, value in iterator:
            check_resources(cancelled)
            if kind == "text":
                if value in {128, 130}:
                    continue
                text_tokens.append(value)
                full = bundle.tokenizer.decode(text_tokens, skip_special_tokens=True)
                if "\ufffd" in full or not full.startswith(last_text):
                    continue
                delta, last_text = full[len(last_text) :], full
                if delta:
                    timings.setdefault("time_to_first_text_seconds", time.perf_counter() - started)
                    yield SpeechStreamEvent(
                        "text_chunk", sequence, text=delta, timings=dict(timings)
                    )
                    sequence += 1
            elif kind == "audio":
                if task == "transcribe":
                    raise ValueError("LFM ASR unexpectedly emitted speech output")
                frames.append(value)
                generated += 1
                if generated > 512:
                    raise ValueError("LFM codec output exceeds 512-frame (40.96-second) bound")
                if len(frames) % chunk_frames == 0:
                    event = decode()
                    if event:
                        yield event
            elif kind in {"audio_end", "end"}:
                event = decode(final=True)
                if event:
                    yield event
                # Independent spans start a fresh codec prefix, retain global PCM offset.
                span_base += emitted
                frames.clear()
                emitted = 0
        timings["wall_seconds"] = time.perf_counter() - started
        if task != "transcribe" and not generated:
            raise ValueError("LFM speech task produced no audio frames")
        yield SpeechStreamEvent("completion", sequence, timings=dict(timings))
    finally:
        iterator.close()
        frames.clear()


def generate_speech(
    model,
    text="",
    *,
    audio=None,
    sample_rate=None,
    voice="us_male",
    revision=None,
    offline=False,
    cache_dir=None,
    **options,
):
    import numpy as np

    from mlx_one.audio.loading import load_audio_model

    task = "speech-to-speech" if audio is not None else "synthesize"
    bundle = (
        model
        if isinstance(model, LoadedLFMAudio)
        else load_audio_model(
            model, task=task, revision=revision, offline=offline, cache_dir=cache_dir
        )
    )
    chunks, words, timings = [], [], {}
    for event in stream_lfm(
        bundle, task=task, text=text, audio=audio, sample_rate=sample_rate, voice=voice, **options
    ):
        if event.kind == "audio_chunk":
            chunks.append(event.waveform)
        elif event.kind == "text_chunk":
            words.append(event.text)
        timings.update(event.timings)
    return SpeechGenerationResult(
        np.concatenate(chunks),
        24000,
        "".join(words),
        str(bundle.path),
        bundle.revision,
        timings,
        {
            "backend": "native-mlx",
            "task": task,
            "codec": "lfm-audio-detokenizer",
            "status": "NATIVE_IMPLEMENTED",
            "codec_streaming": "bounded-prefix-recomputation",
        },
    )


def transcribe_lfm(bundle, audio, *, sample_rate=None, cancelled=None, options=None):
    from mlx_one.audio.input import prepare_audio

    waveform = prepare_audio(audio, sample_rate=sample_rate, max_seconds=180, cancelled=cancelled)
    text, timings = [], {}
    for event in stream_lfm(
        bundle,
        task="transcribe",
        audio=waveform,
        sample_rate=16000,
        cancelled=cancelled,
        max_tokens=getattr(options, "max_tokens", None) or 1024,
    ):
        if event.kind == "text_chunk":
            text.append(event.text)
        timings.update(event.timings)
    return SpeechTranscript(
        "".join(text),
        len(waveform) / 16000,
        str(bundle.path),
        bundle.revision,
        16000,
        timings=timings,
        provenance={
            "backend": "native-mlx",
            "model_type": "lfm2_audio",
            "status": "NATIVE_IMPLEMENTED",
        },
    )
