"""Public native Whisper transcription service."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from mlx_one.audio.decoding import (
    align_words,
    decode_with_fallback,
    detect_language,
    timestamp_segments,
)
from mlx_one.audio.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperDecodeOptions,
)
from mlx_one.models.audio.whisper.loading import LoadedWhisper, load_whisper
from mlx_one.models.audio.whisper.processing import (
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    decode_audio,
    log_mel_spectrogram,
)


def transcribe(
    model: str | Path | LoadedWhisper,
    audio: str | Path | Any,
    *,
    revision: str | None = None,
    language: str | None = None,
    task: Literal["transcribe", "translate"] = "transcribe",
    word_timestamps: bool = False,
    options: WhisperDecodeOptions | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> TranscriptionResult:
    """Transcribe a file or mono 16-kHz waveform with native Whisper."""

    import mlx.core as mx

    started = time.perf_counter()
    bundle = (
        model
        if isinstance(model, LoadedWhisper)
        else load_whisper(model, revision=revision, offline=offline, cache_dir=cache_dir)
    )
    loaded_at = time.perf_counter()
    if isinstance(audio, (str, Path)):
        waveform = decode_audio(audio)
    else:
        waveform = mx.array(audio)
        if waveform.ndim != 1:
            raise ValueError("waveform input must be mono 16-kHz samples")
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
    if opts.suppress_tokens is None and bundle.model.config.suppress_tokens is not None:
        opts = replace(opts, suppress_tokens=tuple(bundle.model.config.suppress_tokens))
    all_tokens: list[int] = []
    segments: list[TranscriptionSegment] = []
    seek = 0
    detected_language = language
    while seek < total_frames:
        chunk = mel[:, seek : seek + N_FRAMES]
        if chunk.shape[1] < N_FRAMES:
            chunk = mx.pad(chunk, ((0, 0), (0, N_FRAMES - chunk.shape[1])))
        encoded, _ = bundle.model.encode(chunk[None, ...])
        mx.eval(encoded)
        if detected_language is None:
            detected_language, _ = detect_language(
                bundle.model, encoded, bundle.tokenizer
            )
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
        for start, end, token_ids in decoded_segments:
            text = bundle.tokenizer.decode(token_ids).strip()
            words = align_words(
                token_ids,
                decoded.cross_attentions,
                bundle.tokenizer,
                alignment_heads=bundle.generation_config.get("alignment_heads"),
                median_width=bundle.model.config.median_filter_width,
                time_offset=offset + start,
                token_logprobs=decoded.token_logprobs,
            ) if word_timestamps else ()
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
        if decoded_segments:
            last_end = decoded_segments[-1][1]
            consumed_frames = round(last_end * SAMPLE_RATE / HOP_LENGTH)
            seek += min(N_FRAMES, max(1, consumed_frames))
        else:
            seek += N_FRAMES
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
