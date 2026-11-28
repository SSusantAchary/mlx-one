"""Click commands for offline or pinned native image inference."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from .checkpoint import convert_checkpoint
from .engine import ImageEngine
from .profiles import IMAGE_PROFILES, image_profile


@click.group("image")
def image_group() -> None:
    """Inspect and run native text-to-image models."""


@image_group.command("list")
def list_images() -> None:
    """Print pinned image profiles and their qualification status."""
    click.echo(json.dumps([asdict(profile) for profile in IMAGE_PROFILES], indent=2))


def _source_options(function):
    function = click.option("--offline", is_flag=True, help="Use the local Hugging Face cache.")(
        function
    )
    function = click.option("--checkpoint", type=click.Path(exists=True, file_okay=False))(function)
    return click.option("--quantization", type=click.Choice(["4", "8"]))(function)


@image_group.command("generate")
@click.argument("model")
@_source_options
@click.option("--prompt", required=True)
@click.option("--width", type=int, default=512, show_default=True)
@click.option("--height", type=int, default=512, show_default=True)
@click.option("--steps", type=int)
@click.option("--seed", type=int)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
def generate_image(
    model: str,
    quantization: str | None,
    checkpoint: str | None,
    offline: bool,
    prompt: str,
    width: int,
    height: int,
    steps: int | None,
    seed: int | None,
    output: Path,
) -> None:
    """Generate one PNG, then print its execution metadata."""
    try:
        with ImageEngine(
            model,
            quantization=int(quantization) if quantization else None,
            checkpoint=checkpoint,
            offline=offline,
        ) as engine:
            result = engine.generate(prompt, width=width, height=height, steps=steps, seed=seed)
        result.save(output)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.metadata, indent=2))


@image_group.command("convert")
@click.argument("model")
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("destination", type=click.Path(file_okay=False, path_type=Path))
@click.option("--quantization", type=click.Choice(["4", "8"]), required=True)
def convert_image(model: str, source: Path, destination: Path, quantization: str) -> None:
    """Export the pinned local checkpoint with MLX affine denoiser weights."""
    try:
        profile = image_profile(model)
        if model == profile.q4_model:
            raise ValueError("the published Q4 checkpoint is ready to use with image generate")
        path = convert_checkpoint(profile, source, destination, bits=int(quantization))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(path))


@image_group.command("serve")
@click.argument("model")
@_source_options
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=8080, show_default=True)
@click.option("--api-key", multiple=True)
@click.option("--idle-ttl", type=click.FloatRange(min=0), default=120, show_default=True)
@click.option("--queue-size", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0), default=300, show_default=True)
def serve_image(
    model: str,
    quantization: str | None,
    checkpoint: str | None,
    offline: bool,
    host: str,
    port: int,
    api_key: tuple[str, ...],
    idle_ttl: float,
    queue_size: int,
    timeout: float,
) -> None:
    """Run the existing OpenAI-compatible server with one on-demand image model."""
    try:
        import uvicorn

        from mlx_one.server import ServerConfig, create_app

        with ImageEngine(
            model,
            quantization=int(quantization) if quantization else None,
            checkpoint=checkpoint,
            offline=offline,
            idle_ttl=idle_ttl,
            queue_size=queue_size,
        ) as engine:
            settings = ServerConfig(api_keys=api_key, warmup=False, timeout=timeout)
            uvicorn.run(
                create_app(config=settings, image_engines={model: engine}), host=host, port=port
            )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
