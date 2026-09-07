"""Command line interface for mlx-one."""

import json
from pathlib import Path

import click

from mlx_one import __version__
from mlx_one.calibration import CalibrationError, run_calibration
from mlx_one.diagnostics import collect_doctor_report, doctor_report_json, format_doctor_report
from mlx_one.hardware import (
    HardwareProfileError,
    detect_hardware,
    format_hardware_profile,
    list_hardware_profiles,
    load_hardware_profile,
)
from mlx_one.inspection import (
    InspectionError,
    format_inspection_result,
    inspect_model,
    write_inspection_result,
)
from mlx_one.planning import (
    PlanningError,
    format_plan_result,
    plan_inference,
    plan_training,
    write_plan_result,
)
from mlx_one.schemas import Precision, TrainingMethod


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="mlx-one")
def main() -> None:
    """Train anywhere. Prove it on Apple Silicon."""


@main.command(help="Check host and MLX backend readiness.")
@click.option("--json-output", "as_json", is_flag=True, help="Print the report as JSON.")
def doctor(as_json: bool) -> None:
    """Report host, Metal, and optional-backend availability."""
    report = collect_doctor_report()
    click.echo(doctor_report_json(report) if as_json else format_doctor_report(report))


@main.command("inspect", help="Inspect model metadata without loading model weights.")
@click.argument("model")
@click.option("--revision", help="Hugging Face branch, tag, or commit to inspect.")
@click.option("--offline", is_flag=True, help="Use only local files or an existing Hub cache.")
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=str),
    help="Hugging Face cache directory.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Print the result as JSON.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=str),
    help="Atomically write the JSON result to this file.",
)
def inspect_command(
    model: str,
    revision: str | None,
    offline: bool,
    cache_dir: str | None,
    as_json: bool,
    output: str | None,
) -> None:
    """Inspect a Hugging Face repository or local model directory."""
    try:
        result = inspect_model(
            model,
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
        )
        saved = write_inspection_result(result, output) if output else None
    except InspectionError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.to_json() if as_json else format_inspection_result(result))
    if saved is not None and not as_json:
        click.echo(f"Saved JSON: {saved}")


@main.group(help="List, inspect, or detect hardware profiles.")
def hardware() -> None:
    """Work with privacy-safe hardware profiles."""


@hardware.command("list", help="List bundled hardware profile identifiers.")
def hardware_list() -> None:
    for profile_id in list_hardware_profiles():
        click.echo(profile_id)


@hardware.command("show", help="Show a built-in or custom hardware profile.")
@click.argument("profile")
@click.option("--json-output", "as_json", is_flag=True, help="Print the profile as JSON.")
def hardware_show(profile: str, as_json: bool) -> None:
    try:
        result = load_hardware_profile(profile)
    except HardwareProfileError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.to_json() if as_json else format_hardware_profile(result))


@hardware.command("detect", help="Detect the current host without stable machine identifiers.")
@click.option("--json-output", "as_json", is_flag=True, help="Print the profile as JSON.")
def hardware_detect(as_json: bool) -> None:
    result = detect_hardware()
    click.echo(result.to_json() if as_json else format_hardware_profile(result))


@main.group(help="Estimate text workload memory without loading model weights.")
def plan() -> None:
    """Build metadata-only inference and training plans."""


def _emit_plan(result, *, as_json: bool, output: str | None) -> None:
    try:
        saved = write_plan_result(result, output) if output else None
    except PlanningError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.to_json() if as_json else format_plan_result(result))
    if saved is not None and not as_json:
        click.echo(f"Saved JSON: {saved}")


@plan.command("inference", help="Plan a text inference workload.")
@click.option("--model", required=True, help="Hugging Face model ID or local directory.")
@click.option("--hardware", "hardware_value", default="local", show_default=True)
@click.option("--revision", help="Hugging Face branch, tag, or commit.")
@click.option("--offline", is_flag=True, help="Use only local or cached metadata.")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=str))
@click.option("--precision", type=click.Choice([item.value for item in Precision]), default="auto")
@click.option("--context-length", type=click.IntRange(min=1), default=2048, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--generation-length", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--json-output", "as_json", is_flag=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str))
def plan_inference_command(
    model: str,
    hardware_value: str,
    revision: str | None,
    offline: bool,
    cache_dir: str | None,
    precision: str,
    context_length: int,
    batch_size: int,
    generation_length: int,
    as_json: bool,
    output: str | None,
) -> None:
    try:
        result = plan_inference(
            model,
            hardware=hardware_value,
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
            precision=precision,
            context_length=context_length,
            batch_size=batch_size,
            generation_length=generation_length,
        )
    except (InspectionError, HardwareProfileError, PlanningError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_plan(result, as_json=as_json, output=output)


@plan.command("train", help="Plan full, LoRA, QLoRA, or scratch text training.")
@click.option("--model", required=True, help="Hugging Face model ID or local directory.")
@click.option("--hardware", "hardware_value", default="local", show_default=True)
@click.option("--revision", help="Hugging Face branch, tag, or commit.")
@click.option("--offline", is_flag=True, help="Use only local or cached metadata.")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=str))
@click.option(
    "--method", type=click.Choice([item.value for item in TrainingMethod]), default="auto"
)
@click.option("--precision", type=click.Choice([item.value for item in Precision]), default="auto")
@click.option("--context-length", type=click.IntRange(min=1), default=2048, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--lora-rank", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--lora-layers", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--gradient-checkpointing", is_flag=True)
@click.option("--optimizer", default="adamw", show_default=True)
@click.option("--json-output", "as_json", is_flag=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str))
def plan_train_command(
    model: str,
    hardware_value: str,
    revision: str | None,
    offline: bool,
    cache_dir: str | None,
    method: str,
    precision: str,
    context_length: int,
    batch_size: int,
    lora_rank: int,
    lora_layers: int,
    gradient_checkpointing: bool,
    optimizer: str,
    as_json: bool,
    output: str | None,
) -> None:
    try:
        result = plan_training(
            model,
            hardware=hardware_value,
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
            method=method,
            precision=precision,
            context_length=context_length,
            batch_size=batch_size,
            lora_rank=lora_rank,
            lora_layers=lora_layers,
            gradient_checkpointing=gradient_checkpointing,
            optimizer=optimizer,
        )
    except (InspectionError, HardwareProfileError, PlanningError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_plan(result, as_json=as_json, output=output)


@main.group(help="Run reproducible hardware calibration matrices.")
def calibrate() -> None:
    """Run reference calibration workloads."""


@calibrate.command("text", help="Run a pinned text inference and PEFT matrix.")
@click.option("--matrix", "matrix_value", required=True)
@click.option("--output", required=True, type=click.Path(file_okay=False, path_type=str))
@click.option(
    "--artifacts-dir",
    default=str(Path.home() / ".cache" / "mlx-one" / "calibration-models"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=str),
)
@click.option("--resume", is_flag=True)
@click.option("--retry", is_flag=True, help="Retry failed or invalid existing cells.")
@click.option("--dry-run", is_flag=True, help="Expand and validate without loading models.")
@click.option("--select", help="Run only cell IDs matching this regular expression.")
def calibrate_text_command(
    matrix_value: str,
    output: str,
    artifacts_dir: str,
    resume: bool,
    retry: bool,
    dry_run: bool,
    select: str | None,
) -> None:
    try:
        summary = run_calibration(
            matrix_value,
            output=output,
            artifacts_dir=artifacts_dir,
            resume=resume,
            retry=retry,
            dry_run=dry_run,
            select=select,
        )
    except CalibrationError as exc:
        raise click.ClickException(str(exc)) from exc
    if dry_run:
        click.echo(f"Matrix: {summary['matrix_id']}")
        click.echo(f"Cells: {summary['cell_count']}")
        for cell in summary["cells"]:
            click.echo(f"  {cell}")
    else:
        click.echo(json.dumps(summary, indent=2, sort_keys=True))


@main.command("eval", help="Run standard model benchmarks.")
def eval_command() -> None:
    """Placeholder for the evaluation harness."""
    click.echo("Unified evaluation — planned for v0.2.0")


@main.command(help="Compare result files and apply quality gates.")
def compare() -> None:
    """Placeholder for result comparison and quality gates."""
    click.echo("Comparison and quality gates — planned for v0.2.0")


@main.command(help="Merge model weights.")
def merge() -> None:
    """Placeholder for model merging."""
    click.echo("MLX-native text model merging — planned for v0.5.0")


@main.command(help="Build and optionally publish a validated release bundle.")
def ship() -> None:
    """Placeholder for release bundles and publishing."""
    click.echo("Quality-gated release bundles — planned for v1.0.0")


if __name__ == "__main__":
    main()
