"""OCR command attached to the existing mlx-one CLI."""

import json
from pathlib import Path

import click

from mlx_one.ocr.documents import parse_page_range
from mlx_one.ocr.schemas import OCRLimits


@click.command("ocr", help="Run native OCR on a local image, TIFF or PDF; no implicit downloads.")
@click.option("--model", required=True)
@click.option("--image", type=click.Path(exists=True, dir_okay=False))
@click.option("--document", type=click.Path(exists=True, dir_okay=False))
@click.option("--task", default=None)
@click.option("--pages", help="One-based inclusive PDF/TIFF range, for example 1-5,8.")
@click.option(
    "--format", "output_format", type=click.Choice(["text", "json", "markdown"]), default="text"
)
@click.option("--output", type=click.Path(dir_okay=False))
@click.option("--dpi", default=144, type=click.IntRange(36, 300))
@click.option("--max-pages", default=100, type=click.IntRange(1, 100))
@click.option("--max-tokens", default=512, type=click.IntRange(1, 16384))
@click.option(
    "--pdf-policy", type=click.Choice(["visual", "embedded_text", "auto"]), default="visual"
)
@click.option("--region", nargs=4, type=int)
@click.option("--revision")
@click.option(
    "--offline", is_flag=True, help="Use local/cache files only (always enforced for OCR)."
)
@click.option("--json", "json_output", is_flag=True)
@click.option("--stream", is_flag=True, help="Emit page-level JSON events, not token deltas.")
def ocr_command(
    model,
    image,
    document,
    task,
    pages,
    output_format,
    output,
    dpi,
    max_pages,
    max_tokens,
    pdf_policy,
    region,
    revision,
    offline,
    json_output,
    stream,
):
    from mlx_one.ocr import OCREngine

    if bool(image) == bool(document):
        raise click.UsageError("provide exactly one of --image or --document")
    if stream and output:
        raise click.UsageError("--output and --stream cannot be combined")
    try:
        with OCREngine(
            model,
            task=task or ("ocr.document" if document else "ocr.recognize"),
            limits=OCRLimits(max_pages=max_pages),
            offline=True,
            revision=revision,
        ) as engine:
            options = dict(
                pages=parse_page_range(pages),
                dpi=dpi,
                max_tokens=max_tokens,
                pdf_policy=pdf_policy,
                region=region or None,
                output_format="json" if json_output else output_format,
            )
            if stream:
                for event in engine.stream(image or document, **options):
                    click.echo(json.dumps(event, ensure_ascii=False))
            else:
                result = engine.recognize(image or document, **options)
                content = result.format(options["output_format"])
                if output:
                    Path(output).expanduser().write_text(content + "\n", encoding="utf-8")
                else:
                    click.echo(content)
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        raise click.ClickException(str(exc)) from exc
