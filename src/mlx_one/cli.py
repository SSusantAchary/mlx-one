"""Command line interface for mlx-one."""

import click

from mlx_one import __version__
from mlx_one.diagnostics import collect_doctor_report, doctor_report_json, format_doctor_report


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
