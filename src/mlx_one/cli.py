"""Command line interface for mlx-one."""

import hashlib
import json
import uuid
from pathlib import Path

import click

from mlx_one import __version__
from mlx_one.benchmark import BenchmarkError, benchmark_inference, load_prompts
from mlx_one.calibration import CalibrationError, run_calibration
from mlx_one.comparison import ComparisonError, compare_results, load_gates, load_result
from mlx_one.datasets import DatasetError, dataset_spec_from_path, load_records, preview_dataset
from mlx_one.diagnostics import collect_doctor_report, doctor_report_json, format_doctor_report
from mlx_one.evaluation import (
    TEXT_EXACT_MATCH_PROFILE,
    TEXT_PROFILES,
    EvaluationError,
    evaluate_text,
    load_evaluation_records,
)
from mlx_one.evidence import EvidenceError, publish_run_evidence
from mlx_one.generation import GenerationError, generate_predictions
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
from mlx_one.registry import CapabilityRegistry, RegistryError
from mlx_one.run_store import RunStore
from mlx_one.schemas import (
    CapabilitySpec,
    CapabilityStatus,
    Modality,
    ModelSpec,
    Operation,
    Precision,
    RunResult,
    RunSpec,
    TrainingMethod,
)
from mlx_one.training import SFTTrainer, TrainingError, export_adapter, load_train_config
from mlx_one.workflow import WorkflowError, run_text_workflow


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


@main.command("eval", help="Run pinned text quality evaluation.")
@click.option("--model")
@click.option("--revision")
@click.option("--dataset", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--predictions", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--adapter", type=click.Path(exists=True, path_type=str))
@click.option(
    "--profile",
    type=click.Choice(TEXT_PROFILES),
    default=TEXT_EXACT_MATCH_PROFILE,
    show_default=True,
)
@click.option("--max-tokens", type=click.IntRange(min=1), default=128)
@click.option("--runs-dir", type=click.Path(file_okay=False, path_type=str), default="runs")
@click.option("--run-id")
@click.option("--json-output", "as_json", is_flag=True)
def eval_command(
    model: str | None,
    revision: str | None,
    dataset: str | None,
    predictions: str | None,
    adapter: str | None,
    profile: str,
    max_tokens: int,
    runs_dir: str,
    run_id: str | None,
    as_json: bool,
) -> None:
    """Evaluate precomputed text outputs under a pinned exact-match protocol."""
    if model is None and revision is None and dataset is None and predictions is None:
        raise click.UsageError("--model, --revision, and --dataset are required")
    if not all((model, revision, dataset)):
        raise click.UsageError("--model, --revision, and --dataset are required")
    try:
        dataset_hash = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
        spec = RunSpec(
            run_id=run_id or f"eval-{uuid.uuid4().hex[:12]}",
            operation=Operation.EVALUATE,
            model=ModelSpec(model_id=model, revision=revision, modality=Modality.TEXT),
            hardware=detect_hardware(),
            profile=profile,
            dataset_revisions={Path(dataset).stem: dataset_hash},
            generation={
                "source": "precomputed" if predictions else "mlx-lm",
                "deterministic": True,
                "max_tokens": max_tokens,
                "adapter": bool(adapter),
            },
        )
        generated = None
        if predictions is None:
            records = load_evaluation_records(dataset)
            generated = generate_predictions(
                model,
                revision,
                [item["prompt"] for item in records],
                max_tokens=max_tokens,
                adapter_path=adapter,
            )
        result = evaluate_text(
            spec,
            dataset,
            predictions=predictions,
            generated_predictions=generated,
            store=RunStore(runs_dir),
        )
    except (EvaluationError, GenerationError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    metric_name = next(name for name in result.metrics if name.startswith("quality."))
    score = result.metrics[metric_name]["value"]
    click.echo(result.to_json() if as_json else f"Run: {result.run_id}\n{metric_name}: {score:.4f}")


main.add_command(eval_command, "evaluate")


@main.command(help="Compare result files and apply quality gates.")
@click.argument("baseline", required=False, type=click.Path(exists=True, dir_okay=False))
@click.argument("candidate", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--gate", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--allow-incompatible", is_flag=True)
@click.option("--format", "output_format", type=click.Choice(["json", "markdown"]), default="json")
@click.option("--output", type=click.Path(dir_okay=False, path_type=str))
def compare(
    baseline: str | None,
    candidate: str | None,
    gate: str | None,
    allow_incompatible: bool,
    output_format: str,
    output: str | None,
) -> None:
    """Compare compatible result files and apply optional quality gates."""
    if baseline is None and candidate is None:
        click.echo("Comparison and quality gates — planned for v0.2.0")
        return
    if baseline is None or candidate is None:
        raise click.UsageError("BASELINE and CANDIDATE are both required")
    try:
        report = compare_results(
            load_result(baseline),
            load_result(candidate),
            gates=load_gates(gate),
            allow_incompatible=allow_incompatible,
        )
    except ComparisonError as exc:
        raise click.ClickException(str(exc)) from exc
    rendered = report.to_json() if output_format == "json" else report.to_markdown()
    if output:
        target = Path(output)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
        temporary.replace(target)
    click.echo(rendered)
    if not report.passed:
        raise click.exceptions.Exit(1)


@main.group(help="Validate local training datasets without starting MLX.")
def data() -> None:
    """Inspect local dataset structure and optional token masks."""


@data.command("validate")
@click.option(
    "--dataset", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.option(
    "--layout",
    default="auto",
    type=click.Choice(["auto", "instruction", "messages", "prompt-completion", "text"]),
)
@click.option("--tokenizer", "tokenizer_ref")
@click.option("--max-seq-length", type=click.IntRange(min=2), default=2048)
def data_validate(
    dataset: str, layout: str, tokenizer_ref: str | None, max_seq_length: int
) -> None:
    try:
        spec = dataset_spec_from_path(dataset, layout=layout)
        records = load_records(spec)
        result: dict[str, object] = {
            "dataset": spec.to_dict(),
            "record_count": len(records),
            "structurally_valid": True,
        }
        if tokenizer_ref:
            from mlx_lm.utils import load_tokenizer

            tokenizer = load_tokenizer(tokenizer_ref)
            result["tokenization"] = preview_dataset(spec, tokenizer, max_length=max_seq_length)
    except (DatasetError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@main.command("train", help="Run native MLX LoRA/QLoRA SFT from a strict configuration.")
@click.option("--model", required=True)
@click.option("--revision", required=True)
@click.option(
    "--dataset", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.option("--resume-from", type=click.Path(exists=True, dir_okay=False, path_type=str))
def train_command(
    model: str, revision: str, dataset: str, config_path: str, resume_from: str | None
) -> None:
    try:
        trainer = SFTTrainer(
            model=model,
            revision=revision,
            train_dataset=dataset,
            args=load_train_config(config_path),
        )
        result = trainer.train(resume_from_checkpoint=resume_from)
    except TrainingError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.to_json())


@main.command("export", help="Verify and export an adapter checkpoint bundle.")
@click.option(
    "--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.option("--output", required=True, type=click.Path(file_okay=False, path_type=str))
@click.option("--smoke-prompt")
@click.option("--max-tokens", type=click.IntRange(min=1), default=16, show_default=True)
def export_command(checkpoint: str, output: str, smoke_prompt: str | None, max_tokens: int) -> None:
    try:
        exported = export_adapter(
            checkpoint, output, smoke_prompt=smoke_prompt, max_tokens=max_tokens
        )
    except TrainingError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Exported adapter bundle: {exported}")


@main.group(help="Inspect the evidence-backed model capability registry.")
def registry() -> None:
    """Query built-in exact-revision support evidence."""


@registry.command("list")
@click.option("--model")
@click.option("--operation", type=click.Choice([item.value for item in Operation]))
def registry_list(model: str | None, operation: str | None) -> None:
    try:
        catalog = CapabilityRegistry.builtin()
        entries = catalog.entries if model is None else catalog.find(model, operation=operation)
    except RegistryError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps([entry.to_dict() for entry in entries], indent=2, sort_keys=True))


@registry.command("validate")
@click.argument("registry_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
def registry_validate(registry_path: str) -> None:
    """Validate every capability in a registry file."""
    try:
        catalog = CapabilityRegistry.from_file(registry_path)
    except (RegistryError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Valid registry: {len(catalog.entries)} entries")


@registry.command("import")
@click.option(
    "--entry", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.option("--registry", "registry_path", required=True, type=click.Path(dir_okay=False))
def registry_import(entry: str, registry_path: str) -> None:
    """Validate and import one candidate capability entry."""
    try:
        target = Path(registry_path)
        catalog = CapabilityRegistry.from_file(target) if target.exists() else CapabilityRegistry()
        payload = json.loads(Path(entry).read_text(encoding="utf-8"))
        catalog.add(CapabilitySpec.from_dict(payload))
        catalog.write(target)
    except (RegistryError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Imported capability into {registry_path}")


@registry.command("promote")
@click.option(
    "--entry", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.option(
    "--result", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.option("--registry", "registry_path", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--status",
    required=True,
    type=click.Choice(
        [
            CapabilityStatus.INTEGRATION_TESTED.value,
            CapabilityStatus.HARDWARE_VERIFIED.value,
            CapabilityStatus.QUALITY_VERIFIED.value,
        ]
    ),
)
def registry_promote(entry: str, result: str, registry_path: str, status: str) -> None:
    """Promote a capability using a matching completed run result."""
    try:
        target = Path(registry_path)
        catalog = CapabilityRegistry.from_file(target) if target.exists() else CapabilityRegistry()
        capability = CapabilitySpec.from_dict(json.loads(Path(entry).read_text(encoding="utf-8")))
        run = RunResult.from_json(Path(result).read_text(encoding="utf-8"))
        promoted = catalog.promote_from_result(capability, run, result, status)
        catalog.write(target)
    except (RegistryError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(promoted.to_json())


@main.group(help="Validate and publish privacy-safe run evidence.")
def evidence() -> None:
    """Work with publishable lifecycle evidence."""


@evidence.command("publish")
@click.option(
    "--result", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str)
)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=str))
def evidence_publish(result: str, output: str) -> None:
    """Sanitize, validate, and checksum one completed run result."""
    try:
        path, digest = publish_run_evidence(result, output)
    except EvidenceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Published evidence: {path}\nSHA-256: {digest}")


@main.group(help="Measure model performance in isolated MLX workers.")
def benchmark() -> None:
    """Run reproducible hardware performance workloads."""


@benchmark.command("inference")
@click.option("--model", required=True)
@click.option("--revision", required=True)
@click.option(
    "--prompts",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.option("--runs-dir", default="runs", type=click.Path(file_okay=False, path_type=str))
@click.option("--max-tokens", type=click.IntRange(min=1), default=32)
@click.option("--repeats", type=click.IntRange(min=1), default=3)
@click.option("--run-id")
def benchmark_inference_command(
    model: str,
    revision: str,
    prompts: str,
    runs_dir: str,
    max_tokens: int,
    repeats: int,
    run_id: str | None,
) -> None:
    try:
        result = benchmark_inference(
            model,
            revision,
            load_prompts(prompts),
            store=RunStore(runs_dir),
            max_tokens=max_tokens,
            repeats=repeats,
            run_id=run_id,
        )
    except BenchmarkError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.to_json())


@main.group(help="Run resumable end-to-end reference workflows.")
def workflow() -> None:
    """Orchestrate complete model lifecycle workflows."""


@workflow.command("text-sft")
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--resume", is_flag=True, help="Skip stages already recorded as completed.")
def workflow_text_sft(config_path: str, resume: bool) -> None:
    """Run inspect, plan, SFT, evaluation, export, benchmark, and comparison."""
    try:
        report = run_text_workflow(config_path, resume=resume)
    except WorkflowError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Workflow completed: {report}")


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
