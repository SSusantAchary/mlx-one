"""Native Qwen offline ASR and known-text alignment, with request-owned caches."""

from __future__ import annotations

import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_one.audio.loading import load_audio_model
from mlx_one.audio.moonshine import BUDGET, AudioResourceLimitError
from mlx_one.audio.schemas import (
    AlignmentUnit,
    ASRDecodeOptions,
    AudioAlignment,
    QwenASROptions,
    SpeechTranscript,
)
from mlx_one.models.audio.qwen3_asr.alignment import fix_timestamps, text_units
from mlx_one.models.audio.qwen3_asr.config import audio_token_length
from mlx_one.models.audio.qwen3_asr.loading import LoadedQwenAudio
from mlx_one.models.audio.qwen3_asr.processing import (
    language_name,
    parse_output,
    preprocess,
    render_prompt,
)


def check_resources(cancelled: Callable[[], bool] | None = None) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("audio inference was cancelled")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
        raise AudioResourceLimitError("24 GiB audio process-memory budget exceeded")


def waveform_input(
    audio: Any, sample_rate: int | None, cancelled: Callable[[], bool] | None
) -> Any:
    is_file = isinstance(audio, (str, Path))
    if sample_rate is not None and (isinstance(sample_rate, bool) or sample_rate != 16000):
        raise ValueError("Qwen waveform sample_rate must be 16000; resample explicitly")
    if not is_file and sample_rate is None:
        raise ValueError("Qwen waveform inputs require an explicit sample_rate=16000")
    check_resources(cancelled)
    import mlx.core as mx

    from mlx_one.models.audio.whisper.processing import decode_audio

    waveform = (
        decode_audio(audio, cancelled=cancelled) if is_file else mx.array(audio, dtype=mx.float32)
    )
    if waveform.ndim != 1 or not mx.all(mx.isfinite(waveform)).item():
        raise ValueError("Qwen waveform must contain finite mono samples")
    if waveform.shape[0]:
        peak = mx.max(mx.abs(waveform)).item()
        if peak > 1.0:
            waveform = mx.clip(waveform / peak, -1.0, 1.0)
    return waveform


def provenance(bundle: LoadedQwenAudio) -> dict[str, Any]:
    return {
        "model_type": bundle.model.config.model_type,
        "checkpoint_path": str(bundle.path),
        "checkpoint_revision": bundle.revision,
        "reference_source_commit": "7c6daf77a2421100f5fb066495372c00129d39ff",
        "audio_attention": "pinned-sdpa-global",
        "decoding": "greedy",
        "weight_dtype": str(bundle.model.audio_tower.conv2d1.weight.dtype),
    }


def preflight(bundle: LoadedQwenAudio, samples: int, prompt_length: int = 0) -> None:
    config = bundle.model.config
    length = audio_token_length(samples // 160)
    text = config.text_config
    if prompt_length > text.max_position_embeddings:
        raise AudioResourceLimitError(
            "Qwen prompt exceeds checkpoint positional context; no truncation"
        )
    working = 5 * 4 * config.audio_config.encoder_attention_heads * length**2
    if working + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
        raise AudioResourceLimitError(
            "Qwen dense audio attention would exceed the 24 GiB process budget"
        )


def decode_qwen(
    bundle: LoadedQwenAudio,
    waveform: Any,
    *,
    language: str | None,
    context: str = "",
    prefix: str = "",
    max_tokens: int = 512,
    offline_padding: bool = True,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[str, tuple[int, ...]]:
    if bundle.model.config.model_type != "qwen3_asr":
        raise ValueError("ForcedAligner cannot transcribe audio")
    import mlx.core as mx

    check_resources(cancelled)
    preflight(bundle, waveform.shape[0])
    features, count = preprocess(waveform, offline_padding=offline_padding)
    prompt = render_prompt(bundle.chat_template, context, language, count, prefix)
    ids = bundle.tokenizer.encode(prompt, add_special_tokens=False).ids
    if ids.count(bundle.model.config.audio_token_id) != count:
        raise ValueError("context/prefix cannot contain audio placeholder special tokens")
    preflight(bundle, waveform.shape[0], len(ids) + max_tokens)
    cache = bundle.model.make_cache()
    tokens: list[int] = []
    try:
        input_ids = mx.array([ids])
        for step in range(max_tokens):
            check_resources(cancelled)
            logits = bundle.model(
                input_ids,
                features=features if step == 0 else None,
                cache=cache,
                output_positions=mx.array([input_ids.shape[1] - 1]),
            )
            token = int(mx.argmax(logits[0, -1]).item())
            if token in bundle.generation_config["eos_token_id"]:
                break
            tokens.append(token)
            input_ids = mx.array([[token]])
        else:
            raise AudioResourceLimitError(
                f"Qwen did not reach EOS within {max_tokens} tokens; refusing truncation"
            )
        check_resources(cancelled)
    finally:
        for state in cache:
            state.reset()
    return prefix + bundle.tokenizer.decode(tokens, skip_special_tokens=True), tuple(tokens)


def transcribe_qwen(
    bundle: LoadedQwenAudio,
    audio: Any,
    *,
    model_name: str,
    sample_rate: int | None,
    language: str | None,
    task: str,
    word_timestamps: bool,
    options: ASRDecodeOptions | None,
    cancelled: Callable[[], bool] | None,
    started: float,
    loaded_at: float,
) -> SpeechTranscript:
    if bundle.model.config.model_type != "qwen3_asr":
        raise ValueError("ForcedAligner cannot transcribe audio; use align_audio")
    if task != "transcribe" or word_timestamps:
        raise ValueError(
            "Qwen ASR supports transcription only; use align_audio for known-text timestamps"
        )
    if options is not None and not isinstance(options, ASRDecodeOptions):
        raise ValueError(
            "Qwen ASR requires ASRDecodeOptions or QwenASROptions, not Whisper options"
        )
    language = language_name(language)
    opts = options or QwenASROptions()
    waveform = waveform_input(audio, sample_rate, cancelled)
    processed = time.perf_counter()
    raw, tokens = (
        decode_qwen(
            bundle,
            waveform,
            language=language,
            context=getattr(opts, "context", ""),
            max_tokens=opts.max_tokens or 512,
            cancelled=cancelled,
        )
        if waveform.shape[0]
        else ("", ())
    )
    detected, text = parse_output(raw, language)
    completed = time.perf_counter()
    return SpeechTranscript(
        text,
        waveform.shape[0] / 16000,
        model_name,
        bundle.revision,
        16000,
        tokens=tokens,
        language=detected,
        timings={
            "load_seconds": loaded_at - started,
            "preprocess_seconds": processed - loaded_at,
            "decode_seconds": completed - processed,
            "wall_seconds": completed - started,
        },
        provenance=provenance(bundle),
    )


def align_audio(
    model: str | Path | LoadedQwenAudio,
    audio: Any,
    text: str,
    *,
    language: str,
    sample_rate: int | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AudioAlignment:
    bundle = (
        model
        if isinstance(model, LoadedQwenAudio)
        else load_audio_model(
            model, task="align", revision=revision, offline=offline, cache_dir=cache_dir
        )
    )
    if (
        not isinstance(bundle, LoadedQwenAudio)
        or bundle.model.config.model_type != "qwen3_forced_aligner"
    ):
        raise ValueError("align_audio requires a native ForcedAligner checkpoint")
    language = language_name(language, bundle.model.config.support_languages)
    if language is None:
        raise ValueError("known-text alignment requires a language")
    words = text_units(text, language, bundle.path)
    waveform = waveform_input(audio, sample_rate, cancelled)
    duration = waveform.shape[0] / 16000
    if duration > 180:
        raise AudioResourceLimitError(
            "pinned Qwen alignment limit is 180 seconds; no implicit segmentation"
        )
    units = ()
    if words and waveform.shape[0]:
        import mlx.core as mx

        preflight(bundle, waveform.shape[0])
        features, count = preprocess(waveform, offline_padding=False)
        prompt = "<|audio_start|>" + "<|audio_pad|>" * count + "<|audio_end|>"
        prompt += "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        ids = bundle.tokenizer.encode(prompt, add_special_tokens=False).ids
        positions = [
            i for i, token in enumerate(ids) if token == bundle.model.config.timestamp_token_id
        ]
        if (
            len(positions) != len(words) * 2
            or ids.count(bundle.model.config.audio_token_id) != count
        ):
            raise ValueError("alignment units contain reserved Qwen special tokens")
        preflight(bundle, waveform.shape[0], len(ids))
        check_resources(cancelled)
        logits = bundle.model(
            mx.array([ids]), features=features, output_positions=mx.array(positions)
        )
        predicted = mx.argmax(logits[0], axis=-1).tolist()
        tick = bundle.model.config.timestamp_segment_time
        boundaries = fix_timestamps([int(value * tick) for value in predicted])
        units = tuple(
            AlignmentUnit(
                word, round(boundaries[i * 2] / 1000, 3), round(boundaries[i * 2 + 1] / 1000, 3)
            )
            for i, word in enumerate(words)
        )
        check_resources(cancelled)
    elif words:
        raise ValueError("cannot align nonempty text to empty audio")
    return AudioAlignment(
        units,
        text,
        language,
        duration,
        str(bundle.path) if isinstance(model, LoadedQwenAudio) else str(model),
        bundle.revision,
        bundle.model.config.timestamp_segment_time / 1000,
        provenance=provenance(bundle),
    )
