"""Model-dispatched native transcription; legacy Whisper results are unchanged."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from mlx_one.audio.decoding import (
    align_words,
    decode_with_fallback,
    detect_language,
    timestamp_seek_frames,
    timestamp_segments,
)
from mlx_one.audio.loading import load_audio_model
from mlx_one.audio.schemas import (
    ASRDecodeOptions,
    QwenASROptions,
    SpeechTranscript,
    TranscriptionResult,
    TranscriptionSegment,
    WhisperDecodeOptions,
)
from mlx_one.models.audio.moonshine.loading import LoadedMoonshine
from mlx_one.models.audio.parakeet.loading import LoadedParakeet
from mlx_one.models.audio.qwen3_asr.loading import LoadedQwenAudio
from mlx_one.models.audio.whisper.loading import LoadedWhisper
from mlx_one.models.audio.whisper.processing import (
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    decode_audio,
    log_mel_spectrogram,
)


def transcribe(
    model: str | Path | LoadedWhisper | LoadedMoonshine | LoadedQwenAudio | LoadedParakeet,
    audio: str | Path | Any,
    *,
    revision: str | None = None,
    language: str | None = None,
    task: Literal["transcribe", "translate"] = "transcribe",
    word_timestamps: bool = False,
    options: WhisperDecodeOptions | ASRDecodeOptions | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cancelled: Callable[[], bool] | None = None,
    sample_rate: int | None = None,
) -> TranscriptionResult | SpeechTranscript:
    """Transcribe with a native family; new waveform APIs require a sample rate.

    Legacy Whisper arrays still implicitly use 16 kHz, and return the original
    TranscriptionResult/JSON schema. Moonshine uses a separate SpeechTranscript.
    """

    from mlx_one.models.audio.lfm2_audio.loading import LoadedLFMAudio

    started = time.perf_counter()
    bundle = (
        model
        if isinstance(
            model, (LoadedWhisper, LoadedMoonshine, LoadedQwenAudio, LoadedParakeet, LoadedLFMAudio)
        )
        else load_audio_model(model, revision=revision, offline=offline, cache_dir=cache_dir)
    )
    loaded_at = time.perf_counter()
    if isinstance(bundle, LoadedLFMAudio):
        from mlx_one.audio.lfm import transcribe_lfm

        if task != "transcribe" or word_timestamps or language is not None:
            raise ValueError("LFM ASR has no translation, timestamp or language-hint API")
        if options is not None and type(options) is not ASRDecodeOptions:
            raise ValueError("LFM ASR accepts ASRDecodeOptions only")
        return transcribe_lfm(
            bundle, audio, sample_rate=sample_rate, cancelled=cancelled, options=options
        )
    if isinstance(bundle, LoadedParakeet):
        from mlx_one.audio.parakeet import transcribe_parakeet

        return transcribe_parakeet(
            bundle,
            audio,
            model_name=str(bundle.path) if isinstance(model, LoadedParakeet) else str(model),
            sample_rate=sample_rate,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            options=options,
            cancelled=cancelled,
            started=started,
            loaded_at=loaded_at,
        )
    if isinstance(bundle, LoadedQwenAudio):
        from mlx_one.audio.qwen import transcribe_qwen

        return transcribe_qwen(
            bundle,
            audio,
            model_name=str(bundle.path) if isinstance(model, LoadedQwenAudio) else str(model),
            sample_rate=sample_rate,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            options=options,
            cancelled=cancelled,
            started=started,
            loaded_at=loaded_at,
        )
    if isinstance(bundle, LoadedMoonshine):
        if isinstance(options, QwenASROptions):
            raise ValueError("Moonshine does not support Qwen context/decoding options")
        from mlx_one.audio.moonshine import transcribe_moonshine

        return transcribe_moonshine(
            bundle,
            audio,
            model_name=str(bundle.path) if isinstance(model, LoadedMoonshine) else str(model),
            sample_rate=sample_rate,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            options=options,
            cancelled=cancelled,
            started=started,
            loaded_at=loaded_at,
        )
    if not isinstance(bundle, LoadedWhisper):
        raise ValueError("audio loader returned an unsupported native transcription bundle")
    if options is not None and not isinstance(options, WhisperDecodeOptions):
        raise ValueError("Whisper requires WhisperDecodeOptions")
    if sample_rate is not None and (isinstance(sample_rate, bool) or sample_rate != SAMPLE_RATE):
        raise ValueError("Whisper waveform sample_rate must be 16000")
    _raise_if_cancelled(cancelled)
    import mlx.core as mx

    if isinstance(audio, (str, Path)):
        waveform = decode_audio(audio, cancelled=cancelled)
    else:
        waveform = mx.array(audio)
        if waveform.ndim != 1:
            raise ValueError("waveform input must be mono 16-kHz samples")
    if not mx.all(mx.isfinite(waveform)).item():
        raise ValueError("Whisper waveform samples must be finite")
    if waveform.size == 0:
        return TranscriptionResult(
            "", language or "en", (), 0.0, str(bundle.path), bundle.revision, task=task
        )
    duration = waveform.shape[0] / SAMPLE_RATE
    filters = bundle.processor.mel_filters
    mel = log_mel_spectrogram(
        waveform,
        num_mel_bins=bundle.model.config.num_mel_bins,
        mel_filters=filters,
        padding=N_SAMPLES,
    )
    mx.eval(mel)
    processed_at = time.perf_counter()
    total_frames = max(1, mel.shape[1] - N_FRAMES)
    opts = options or WhisperDecodeOptions()
    if opts.suppress_tokens is None:
        suppressed = bundle.generation_config.get(
            "suppress_tokens", bundle.model.config.suppress_tokens
        )
        if suppressed is not None:
            opts = replace(opts, suppress_tokens=tuple(suppressed))
    all_tokens: list[int] = []
    segments: list[TranscriptionSegment] = []
    seek = 0
    detected_language = language
    while seek < total_frames:
        _raise_if_cancelled(cancelled)
        chunk = mel[:, seek : seek + N_FRAMES]
        if chunk.shape[1] < N_FRAMES:
            chunk = mx.pad(chunk, ((0, 0), (0, N_FRAMES - chunk.shape[1])))
        encoded, _ = bundle.model.encode(chunk[None, ...])
        mx.eval(encoded)
        if detected_language is None:
            detected_language, _ = detect_language(bundle.model, encoded, bundle.tokenizer)
        prompt = None
        if opts.initial_prompt and not all_tokens:
            prompt = bundle.tokenizer.encode(" " + opts.initial_prompt.strip())
        elif opts.condition_on_previous_text and all_tokens:
            prompt = all_tokens[-bundle.model.config.max_target_positions // 2 :]
        decoded = decode_with_fallback(
            bundle.model,
            encoded,
            bundle.tokenizer,
            language=detected_language,
            task=task,
            word_timestamps=word_timestamps,
            options=opts,
            prompt_tokens=prompt,
            cancelled=cancelled,
        )
        if (
            opts.no_speech_threshold is not None
            and decoded.no_speech_prob > opts.no_speech_threshold
            and (opts.logprob_threshold is None or decoded.avg_logprob < opts.logprob_threshold)
        ):
            seek += N_FRAMES
            continue
        offset = seek * HOP_LENGTH / SAMPLE_RATE
        decoded_segments = timestamp_segments(decoded.tokens, bundle.tokenizer)
        if not decoded_segments and decoded.text:
            decoded_segments = [(0.0, min(30.0, duration - offset), decoded.tokens)]
        token_cursor = 0
        for start, end, token_ids in decoded_segments:
            # Find this segment's own generated rows, retaining repeated-word order.
            token_start = next(
                (
                    index
                    for index in range(token_cursor, len(decoded.tokens))
                    if decoded.tokens[index : index + len(token_ids)] == token_ids
                ),
                token_cursor,
            )
            token_cursor = token_start + len(token_ids)
            segment_end = min(end, duration - offset)
            if start > segment_end:
                raise ValueError("Whisper predicted a segment outside the available audio")
            text = bundle.tokenizer.decode(token_ids).strip()
            words = (
                align_words(
                    token_ids,
                    decoded.cross_attentions,
                    bundle.tokenizer,
                    alignment_heads=bundle.generation_config.get("alignment_heads"),
                    median_width=bundle.model.config.median_filter_width,
                    time_offset=offset + start,
                    token_logprobs=decoded.token_logprobs[token_start:token_cursor],
                    token_start=token_start,
                    frame_start=round(start / 0.02),
                    frame_end=max(round(start / 0.02) + 1, round(segment_end / 0.02)),
                )
                if word_timestamps
                else ()
            )
            segments.append(
                TranscriptionSegment(
                    id=len(segments),
                    seek=seek,
                    start=offset + start,
                    end=min(duration, offset + end),
                    text=text,
                    tokens=token_ids,
                    temperature=decoded.temperature,
                    avg_logprob=decoded.avg_logprob,
                    compression_ratio=decoded.compression_ratio,
                    no_speech_prob=decoded.no_speech_prob,
                    words=words,
                )
            )
            all_tokens.extend(token_ids)
        seek += timestamp_seek_frames(
            decoded.tokens, bundle.tokenizer, min(N_FRAMES, total_frames - seek)
        )
    completed_at = time.perf_counter()
    model_name = str(model) if not isinstance(model, LoadedWhisper) else str(bundle.path)
    return TranscriptionResult(
        text=" ".join(segment.text for segment in segments if segment.text).strip(),
        language=detected_language or "en",
        segments=tuple(segments),
        duration=duration,
        model=model_name,
        revision=revision or bundle.revision,
        task=task,
        timings={
            "load_seconds": loaded_at - started,
            "preprocess_seconds": processed_at - loaded_at,
            "decode_seconds": completed_at - processed_at,
            "wall_seconds": completed_at - started,
        },
    )


def _raise_if_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("transcription was cancelled")


def detect_audio_language(bundle, audio, *, sample_rate=16000, cancelled=None):
    """Whisper encoder language probabilities, or Qwen's generated language label."""
    from mlx_one.audio.input import prepare_audio

    pcm = prepare_audio(audio, sample_rate=sample_rate, max_seconds=180, cancelled=cancelled)
    if not len(pcm):
        raise ValueError("language detection requires nonempty audio")
    if isinstance(bundle, LoadedQwenAudio) and bundle.model.config.model_type == "qwen3_asr":
        result = transcribe(bundle, pcm, sample_rate=16000, cancelled=cancelled)
        return {
            "language": result.language,
            "probabilities": None,
            "model": result.model,
            "revision": result.revision,
            "method": "generated-asr-language-label",
        }
    if not isinstance(bundle, LoadedWhisper) or "<|en|>" not in bundle.tokenizer.special_tokens:
        raise ValueError("language detection requires multilingual Whisper or Qwen ASR")
    import mlx.core as mx

    _raise_if_cancelled(cancelled)
    waveform = mx.array(pcm[:N_SAMPLES])
    mel = log_mel_spectrogram(
        waveform,
        num_mel_bins=bundle.model.config.num_mel_bins,
        mel_filters=bundle.processor.mel_filters,
        padding=N_SAMPLES,
    )
    mel = mel[:, :N_FRAMES]
    encoded, _ = bundle.model.encode(mel[None])
    language, probabilities = detect_language(bundle.model, encoded, bundle.tokenizer)
    _raise_if_cancelled(cancelled)
    return {
        "language": language,
        "probabilities": probabilities,
        "model": str(bundle.path),
        "revision": bundle.revision,
        "method": "whisper-first-window",
    }
