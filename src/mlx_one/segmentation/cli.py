"""Click integration; architecture imports remain deferred until inference."""

import json
from pathlib import Path

import click


def _prompts(path, video=False):
    if path is None:
        return [] if video else {}
    payload = json.loads(Path(path).read_text())
    if video:
        if not isinstance(payload, list):
            raise ValueError("video prompt JSON must be an array of timed prompt records")
        allowed = {
            "frame_index",
            "action",
            "text",
            "prompt_id",
            "object_id",
            "points",
            "point_labels",
            "box",
            "mask_path",
        }
        for record in payload:
            if not isinstance(record, dict) or set(record) - allowed:
                raise ValueError("unknown video prompt fields")
            if (
                not isinstance(record.get("frame_index", 0), int)
                or record.get("frame_index", 0) < 0
            ):
                raise ValueError("frame_index must be a nonnegative integer")
            if record.get("action", "add") not in {"add", "update", "remove"}:
                raise ValueError("invalid prompt action")
        return payload
    if not isinstance(payload, dict) or set(payload) - {
        "text",
        "boxes",
        "box_labels",
        "points",
        "point_labels",
        "mask_path",
    }:
        raise ValueError("unknown image prompt fields")
    return payload


def _mask(path, boolean=False):
    if path is None:
        return None
    import numpy as np
    from PIL import Image

    if Path(path).suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            value = np.asarray(image.convert("L"))
        value = value > 0 if boolean else np.where(value > 0, 10.0, -10.0).astype(np.float32)
    if boolean and value.dtype != bool:
        raise ValueError("video .npy masks must be boolean")
    return value


def _apply(session, index, records):
    for record in records:
        if record.get("frame_index", 0) != index:
            continue
        action = record.get("action", "add")
        if action == "remove":
            if "object_id" in record:
                session.remove_object(record["object_id"])
            elif "prompt_id" in record:
                session.remove_prompt(record["prompt_id"])
            else:
                raise ValueError("remove requires an object_id or prompt_id")
        elif "object_id" in record:
            session.correct_object(
                record["object_id"],
                index,
                points=record.get("points"),
                point_labels=record.get("point_labels"),
                box=record.get("box"),
                mask=_mask(record.get("mask_path"), boolean=True),
                prompt_id=record.get("prompt_id"),
            )
        elif "text" in record:
            if action == "update":
                if "prompt_id" not in record:
                    raise ValueError("text update requires prompt_id")
                session.remove_prompt(record["prompt_id"])
            session.add_prompt(record["text"])
        else:
            raise ValueError("add/update requires text or an object-specific spatial prompt")


@click.command("segment", help="Segment an image with native SAM3 (experimental, unqualified).")
@click.argument("model")
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(["concept", "interactive", "automatic"]),
    default="concept",
    show_default=True,
)
@click.option("--text")
@click.option("--prompts-json", type=click.Path(exists=True, dir_okay=False))
@click.option("--threshold", type=click.FloatRange(0, 1), default=0.3, show_default=True)
@click.option("--points-per-side", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--revision")
@click.option("--offline", is_flag=True)
@click.option("--cache-dir", type=click.Path(file_okay=False))
@click.option("--output", required=True, type=click.Path())
def segment_command(
    model,
    image,
    mode,
    text,
    prompts_json,
    threshold,
    points_per_side,
    revision,
    offline,
    cache_dir,
    output,
):
    try:
        from .export import ResultWriter
        from .inference import generate_masks, predict_masks, segment_image
        from .loading import load_segmentation_model

        if Path(output).exists():
            raise FileExistsError("output already exists; overwriting is refused")
        prompts = _prompts(prompts_json)
        bundle = load_segmentation_model(
            model,
            task="concept" if mode == "concept" else "interactive",
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
        )
        if mode == "concept":
            if set(prompts) & {"points", "point_labels", "mask_path"}:
                raise ValueError(
                    "concept mode accepts text and box exemplars, not interactive points/masks"
                )
            result = segment_image(
                bundle,
                image,
                text=text or prompts.get("text"),
                boxes=prompts.get("boxes"),
                box_labels=prompts.get("box_labels"),
                threshold=threshold,
            )
        elif mode == "interactive":
            if text is not None or set(prompts) & {"text", "box_labels"}:
                raise ValueError(
                    "interactive mode does not accept concept text or box exemplar labels"
                )
            result = predict_masks(
                bundle,
                image,
                points=prompts.get("points"),
                point_labels=prompts.get("point_labels"),
                boxes=prompts.get("boxes"),
                mask=_mask(prompts.get("mask_path")),
            )
        else:
            if text is not None or prompts:
                raise ValueError("automatic mode does not accept prompts")
            result = generate_masks(bundle, image, points_per_side=points_per_side)
        writer = ResultWriter(output)
        writer.add(result)
        writer.complete()
        click.echo(str(writer.root / "manifest.json"))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.command(
    "track", help="Track SAM3 masks through local video/frames (experimental, unqualified)."
)
@click.argument("model")
@click.argument("video", type=click.Path(exists=True))
@click.option("--text", multiple=True)
@click.option("--prompts-json", type=click.Path(exists=True, dir_okay=False))
@click.option("--streaming", is_flag=True)
@click.option("--revision")
@click.option("--offline", is_flag=True)
@click.option("--cache-dir", type=click.Path(file_okay=False))
@click.option("--output", required=True, type=click.Path())
def track_command(
    model, video, text, prompts_json, streaming, revision, offline, cache_dir, output
):
    try:
        from .export import ResultWriter
        from .loading import load_segmentation_model
        from .video import iter_frames

        if Path(output).exists():
            raise FileExistsError("output already exists; overwriting is refused")
        prompts = _prompts(prompts_json, video=True)
        if not text and not any(
            p.get("frame_index", 0) == 0 and p.get("action", "add") != "remove" for p in prompts
        ):
            raise ValueError("tracking requires a text or spatial prompt at frame zero")
        bundle = load_segmentation_model(
            model, task="video", revision=revision, offline=offline, cache_dir=cache_dir
        )
        writer = ResultWriter(output)
        with bundle.video_session(None if streaming else video, streaming=streaming) as session:
            for prompt in text:
                session.add_prompt(prompt)

            def callback(s, i):
                _apply(s, i, prompts)

            if streaming:
                iterator = iter_frames(video)
                try:
                    for frame in iterator:
                        writer.add(session.process_frame(frame, before_frame=callback))
                finally:
                    if hasattr(iterator, "close"):
                        iterator.close()
            else:
                _apply(session, 0, prompts)

                def callback(s, i):
                    if i:
                        _apply(s, i, prompts)

                for result in session.propagate(before_frame=callback):
                    writer.add(result)
        writer.complete()
        click.echo(str(writer.root / "manifest.json"))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
