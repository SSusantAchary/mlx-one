"""Audio CLI subcommands; all inference uses the existing managed task service."""

from __future__ import annotations

import json
from pathlib import Path

import click

from mlx_one.audio.public import AudioEngine


@click.group("audio")
def audio_group():
    """Native transcription, alignment and speech generation."""


def common(function):
    for decorator in reversed(
        [
            click.option("--model", required=True, help="Local checkpoint directory or Hub ID."),
            click.option("--revision"),
            click.option("--offline", is_flag=True),
            click.option("--cache-dir", type=click.Path(file_okay=False)),
        ]
    ):
        function = decorator(function)
    return function


def emit(result, output):
    content = json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if output:
        with Path(output).open("x", encoding="utf-8") as handle:
            handle.write(content)
    else:
        click.echo(content, nl=False)


@audio_group.command("transcribe")
@common
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--language")
@click.option("--word-timestamps", is_flag=True)
@click.option("--stream", is_flag=True, help="Qwen accumulated utterance, maximum 30 seconds.")
@click.option("--output", type=click.Path(dir_okay=False))
def transcribe_command(
    model, revision, offline, cache_dir, audio, language, word_timestamps, stream, output
):
    try:
        if output and Path(output).exists():
            raise ValueError("output already exists")
        with AudioEngine(model, revision=revision, offline=offline, cache_dir=cache_dir) as engine:
            if stream:
                if word_timestamps or output:
                    raise ValueError("stream emits NDJSON to stdout; timestamps/output unsupported")
                from mlx_one.audio.input import prepare_audio

                pcm = prepare_audio(audio, max_seconds=30)
                for event in engine.stream_transcribe(
                    (pcm[i : i + 32000] for i in range(0, len(pcm), 32000)), language=language
                ):
                    click.echo(event.to_json())
            else:
                emit(
                    engine.transcribe(audio, language=language, word_timestamps=word_timestamps),
                    output,
                )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


@audio_group.command("align")
@common
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--text", required=True)
@click.option("--language", required=True)
@click.option("--output", type=click.Path(dir_okay=False))
def align_command(model, revision, offline, cache_dir, audio, text, language, output):
    try:
        if output and Path(output).exists():
            raise ValueError("output already exists")
        with AudioEngine(model, revision=revision, offline=offline, cache_dir=cache_dir) as engine:
            emit(engine.align(audio, text, language=language), output)
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


@audio_group.command("speak")
@common
@click.option("--text", default="")
@click.option(
    "--audio",
    type=click.Path(exists=True, dir_okay=False),
    help="Use interleaved speech-to-speech instead of TTS.",
)
@click.option(
    "--voice",
    type=click.Choice(["us_male", "us_female", "uk_male", "uk_female"]),
    default="us_male",
)
@click.option("--max-tokens", type=click.IntRange(min=1, max=4096), default=1024)
@click.option("--format", type=click.Choice(["wav", "pcm"]), default="wav")
@click.option("--output", required=True, type=click.Path(dir_okay=False))
def speak_command(
    model, revision, offline, cache_dir, text, audio, voice, max_tokens, format, output
):
    try:
        if Path(output).exists():
            raise ValueError("output already exists")
        if bool(text.strip()) == bool(audio):
            raise ValueError("provide either --text (TTS) or --audio (speech-to-speech)")
        with AudioEngine(model, revision=revision, offline=offline, cache_dir=cache_dir) as engine:
            result = (
                engine.speech_to_speech(audio, max_tokens=max_tokens)
                if audio
                else engine.generate_speech(text, voice=voice, max_tokens=max_tokens)
            )
            result.save(output, format=format)
            click.echo(json.dumps(result.to_dict(), ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


@audio_group.command("detect-language")
@common
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False))
def detect_language_command(model, revision, offline, cache_dir, audio):
    try:
        with AudioEngine(model, revision=revision, offline=offline, cache_dir=cache_dir) as engine:
            click.echo(json.dumps(engine.detect_language(audio), ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
