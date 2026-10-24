"""Native audio services and routes attached to the existing server scheduler."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from mlx_one.audio.input import FORMATS, AudioInput, prepare_audio
from mlx_one.audio.loading import audio_model_type, load_audio_model
from mlx_one.server.transcription import MAX_AUDIO_BYTES, AudioUploadError


@dataclass(frozen=True)
class AudioServingBundle:
    audio: object
    model_id: str
    architecture: str
    quantization: dict
    revision: str | None
    context_length: int = 8192
    parameter_count: int | None = None
    chat_template: object = None
    modalities: tuple = ("audio",)

    @property
    def resident_bytes(self):
        from mlx_one.engine.cache import array_nbytes

        codec = getattr(self.audio, "codec", None)
        return array_nbytes(self.audio.model.parameters()) + (
            array_nbytes(codec.parameters()) + array_nbytes(getattr(codec, "_window", None))
            if codec is not None
            else 0
        )

    @property
    def model(self):
        return self.audio.model

    @property
    def tokenizer(self):
        return self.audio.tokenizer


def serving_audio(model, root, **options):
    payload = json.loads((root / "config.json").read_text())
    family = audio_model_type(payload)
    bundle = load_audio_model(
        model, task="align" if family == "qwen3_forced_aligner" else "transcribe", **options
    )
    quant = payload.get("quantization", payload.get("quantization_config", {})) or {}
    return AudioServingBundle(bundle, str(model), family, quant, bundle.revision)


class NativeAudioService:
    def __init__(self, bundle):
        self.bundle = bundle
        self.family = audio_model_type({"model_type": bundle.model.config.model_type})

    def close(self):
        self.bundle = None

    def capabilities(self):
        return {
            "enabled": True,
            "accepted_formats": sorted(FORMATS),
            "max_bytes": MAX_AUDIO_BYTES,
            "max_duration_seconds": 180,
            "language_detection": self.family in {"whisper", "qwen3_asr"},
            "streaming": "accumulated-utterance" if self.family == "qwen3_asr" else False,
            "timestamps": ["word", "segment"]
            if self.family == "whisper"
            else ["token"]
            if self.family == "parakeet_tdt"
            else [],
            "speech_generation": self.family == "lfm2_audio",
        }

    def pcm(self, audio, media_type, cancel):
        if media_type.partition(";")[0].lower() not in FORMATS:
            raise AudioUploadError("unsupported audio content type")
        return prepare_audio(
            AudioInput(audio, media_type=media_type),
            max_seconds=180,
            cancelled=cancel.is_set if cancel else None,
        )

    def __call__(
        self,
        audio,
        *,
        media_type,
        language=None,
        task="transcribe",
        cancel=None,
        word_timestamps=False,
    ):
        from mlx_one.audio import transcribe

        pcm = self.pcm(audio, media_type, cancel)
        return transcribe(
            self.bundle,
            pcm,
            sample_rate=16000,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            cancelled=cancel.is_set if cancel else None,
        )

    def detect_language(self, audio, *, media_type, cancel=None):
        from mlx_one.audio.service import detect_audio_language

        return detect_audio_language(
            self.bundle,
            self.pcm(audio, media_type, cancel),
            sample_rate=16000,
            cancelled=cancel.is_set if cancel else None,
        )

    def align(self, audio, *, media_type, text, language, cancel=None):
        from mlx_one.audio import align_audio

        return align_audio(
            self.bundle,
            self.pcm(audio, media_type, cancel),
            text,
            language=language,
            sample_rate=16000,
            cancelled=cancel.is_set if cancel else None,
        )

    def speak(
        self, *, text="", audio=None, media_type="", voice="us_male", max_tokens=1024, cancel=None
    ):
        from mlx_one.audio.lfm import generate_speech

        pcm = self.pcm(audio, media_type, cancel) if audio is not None else None
        return generate_speech(
            self.bundle,
            text,
            audio=pcm,
            sample_rate=16000,
            voice=voice,
            max_tokens=max_tokens,
            cancelled=cancel.is_set if cancel else None,
        )

    def stream(self, audio, *, media_type, language=None, cancel):
        from mlx_one.audio import open_audio_stream

        pcm = self.pcm(audio, media_type, cancel)
        with open_audio_stream(
            self.bundle, sample_rate=16000, language=language, cancelled=cancel.is_set
        ) as session:
            for start in range(0, len(pcm), 32000):
                yield from session.push(pcm[start : start + 32000])
            yield session.finish()

    def stream_speech(
        self, *, text="", audio=None, media_type="", voice="us_male", max_tokens=1024, cancel
    ):
        from mlx_one.audio.lfm import stream_lfm

        pcm = self.pcm(audio, media_type, cancel) if audio is not None else None
        yield from stream_lfm(
            self.bundle,
            task="speech-to-speech" if pcm is not None else "synthesize",
            text=text,
            audio=pcm,
            sample_rate=16000,
            voice=voice,
            max_tokens=max_tokens,
            cancelled=cancel.is_set,
        )


class AudioRequestTooLarge(ValueError):
    pass


def install_audio_routes(app, manager, jobs, services, auth_dependency):
    from mlx_one.server.app import (
        ClientDisconnected,
        _call_service,
        _error,
        _until_disconnected,
        _validate_task_model,
    )
    from mlx_one.server.scheduler import QueueFullError, RequestTimeoutError

    def failure(exc):
        if isinstance(exc, AudioRequestTooLarge):
            return _error(413, str(exc), "invalid_request_error")
        if isinstance(exc, QueueFullError):
            return _error(429, "task queue is full", "rate_limit_error")
        if isinstance(exc, RequestTimeoutError):
            return _error(408, str(exc), "timeout_error")
        if isinstance(exc, AudioUploadError):
            return _error(415, str(exc), "invalid_request_error")
        if isinstance(exc, ClientDisconnected):
            return _error(499, "client disconnected", "cancelled_error")
        if isinstance(exc, (ValueError, OSError)):
            return _error(400, str(exc), "invalid_request_error")
        return _error(500, str(exc), "server_error")

    async def read_upload(form):
        uploaded = form.get("file")
        if uploaded is None or not hasattr(uploaded, "read"):
            raise ValueError("file is required")
        data = await uploaded.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise AudioRequestTooLarge("audio file exceeds 25 MiB")
        return data, str(getattr(uploaded, "content_type", "") or "")

    async def streaming(request, factory, *, speech=False):
        async def frames():
            sequence = 0
            try:
                async for event in _until_disconnected(request, jobs.schedule(factory)):
                    if speech:
                        record = {
                            "kind": event.kind,
                            "sequence": event.sequence,
                            "sample_rate": event.sample_rate,
                            "sample_offset": event.sample_offset,
                            "text": event.text,
                            "timings": event.timings,
                        }
                        if event.waveform is not None:
                            import numpy as np

                            record["pcm_s16le_base64"] = base64.b64encode(
                                (np.clip(event.waveform, -1, 1) * 32767).astype("<i2").tobytes()
                            ).decode("ascii")
                        yield json.dumps(record) + "\n"
                    else:
                        yield "data: " + event.to_json() + "\n\n"
                        if event.kind == "final":
                            yield (
                                "data: "
                                + json.dumps({"kind": "completion", "sequence": event.sequence + 1})
                                + "\n\n"
                            )
                    sequence = event.sequence + 1
            except ClientDisconnected:
                return
            except Exception as exc:
                record = json.dumps({"kind": "error", "sequence": sequence, "message": str(exc)})
                yield record + "\n" if speech else "data: " + record + "\n\n"

        return StreamingResponse(
            frames(), media_type="application/x-ndjson" if speech else "text/event-stream"
        )

    @app.post("/v1/audio/transcriptions")
    @app.post("/v1/audio/translations")
    async def transcriptions(request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        service = services.get("transcription")
        if service is None:
            return _error(
                400, "the loaded model does not support transcription", "capability_error"
            )
        try:
            async with request.form() as form:
                invalid = _validate_task_model(manager, str(form.get("model", "")))
                if invalid is not None:
                    return invalid
                format = str(form.get("response_format", "json"))
                if format not in {"json", "verbose_json", "text"}:
                    raise ValueError("response_format must be json, verbose_json or text")
                audio, media = await read_upload(form)
                task = (
                    "translate"
                    if request.url.path.endswith("translations")
                    else str(form.get("task", "transcribe"))
                )
                if task == "translate" and getattr(service, "family", "whisper") != "whisper":
                    raise ValueError("translation is supported by multilingual Whisper only")
                language = form.get("language")
                if str(form.get("stream", "false")).lower() == "true":
                    if getattr(service, "family", None) != "qwen3_asr" or task != "transcribe":
                        raise ValueError("transcription streaming requires Qwen ASR")
                    return await streaming(
                        request,
                        lambda cancel: service.stream(
                            audio, media_type=media, language=language, cancel=cancel
                        ),
                    )
                kwargs = {"media_type": media, "language": language, "task": task}
                granularities = form.getlist("timestamp_granularities[]")
                if any(g not in {"word", "segment"} for g in granularities):
                    raise ValueError("unsupported timestamp granularity")
                if granularities:
                    if getattr(service, "family", "whisper") != "whisper":
                        raise ValueError("word/segment timestamps require Whisper")
                    kwargs["word_timestamps"] = "word" in granularities
                result = await _call_service(jobs, service, audio, request=request, **kwargs)
            payload = result.to_dict() if hasattr(result, "to_dict") else result
            if format == "text":
                return Response(payload.get("text", ""), media_type="text/plain")
            if format == "verbose_json":
                return payload
            return {
                "text": payload.get("text", ""),
                **({"language": payload["language"]} if payload.get("language") else {}),
                "mlx": {
                    k: v
                    for k, v in payload.items()
                    if k in {"duration", "segments", "timings", "timestamps", "provenance"}
                },
            }
        except Exception as exc:
            return failure(exc)

    @app.post("/v1/audio/language")
    async def language_detection(request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        service = services.get("transcription")
        if getattr(service, "family", None) not in {"whisper", "qwen3_asr"}:
            return _error(400, "model cannot detect audio language", "capability_error")
        try:
            async with request.form() as form:
                invalid = _validate_task_model(manager, str(form.get("model", "")))
                if invalid is not None:
                    return invalid
                audio, media = await read_upload(form)
                return await _call_service(
                    jobs, service.detect_language, audio, media_type=media, request=request
                )
        except Exception as exc:
            return failure(exc)

    @app.post("/v1/audio/alignments")
    async def alignments(request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        service = services.get("alignment")
        if service is None:
            return _error(400, "the loaded model cannot align known text", "capability_error")
        try:
            async with request.form() as form:
                invalid = _validate_task_model(manager, str(form.get("model", "")))
                if invalid is not None:
                    return invalid
                text, language = form.get("text"), form.get("language")
                if not isinstance(text, str) or not isinstance(language, str):
                    raise ValueError("alignment requires text and language")
                audio, media = await read_upload(form)
                result = await _call_service(
                    jobs,
                    service.align,
                    audio,
                    media_type=media,
                    text=text,
                    language=language,
                    request=request,
                )
                return result.to_dict()
        except Exception as exc:
            return failure(exc)

    @app.post("/v1/audio/speech")
    @app.post("/v1/audio/speech-to-speech")
    async def speech(request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        service = services.get("speech")
        if service is None:
            return _error(400, "the loaded model cannot generate speech", "capability_error")
        try:
            if request.url.path.endswith("speech-to-speech"):
                async with request.form() as form:
                    body = dict(form)
                    data, media = await read_upload(form)
                kwargs = {"audio": data, "media_type": media}
            else:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("speech body must be a JSON object")
                kwargs = {"text": body.get("input", "")}
            invalid = _validate_task_model(manager, str(body.get("model", "")))
            if invalid is not None:
                return invalid
            if body.get("speed", 1.0) != 1.0:
                raise ValueError("LFM audio does not implement speech speed control")
            format = body.get("response_format", "wav")
            if format not in {"wav", "pcm"}:
                raise ValueError("speech response_format must be wav or pcm")
            kwargs.update(
                voice=body.get("voice", "us_male"), max_tokens=int(body.get("max_tokens", 1024))
            )
            if str(body.get("stream", False)).lower() == "true":
                return await streaming(
                    request,
                    lambda cancel: service.stream_speech(cancel=cancel, **kwargs),
                    speech=True,
                )
            result = await _call_service(jobs, service.speak, request=request, **kwargs)
            return Response(
                result.audio_bytes(format),
                media_type="audio/wav" if format == "wav" else "application/octet-stream",
                headers={"X-Audio-Sample-Rate": str(result.sample_rate), "X-Audio-Channels": "1"},
            )
        except Exception as exc:
            return failure(exc)
