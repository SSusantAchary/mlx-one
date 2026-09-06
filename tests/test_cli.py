from click.testing import CliRunner

from mlx_one import __version__
from mlx_one.cli import main
from mlx_one.diagnostics import BackendStatus, DoctorReport


def test_version() -> None:
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    for command in ("doctor", "eval", "compare", "merge", "ship"):
        assert command in result.output
    for deferred_command in ("data", "export", "pipeline"):
        assert deferred_command not in result.output


def test_eval_placeholder() -> None:
    result = CliRunner().invoke(main, ["eval"])

    assert result.exit_code == 0
    assert result.output.strip() == "Unified evaluation — planned for v0.2.0"


def test_all_placeholder_commands() -> None:
    runner = CliRunner()
    expected = {
        "compare": "Comparison and quality gates — planned for v0.2.0",
        "merge": "MLX-native text model merging — planned for v0.5.0",
        "ship": "Quality-gated release bundles — planned for v1.0.0",
    }

    for command, output in expected.items():
        result = runner.invoke(main, [command])
        assert result.exit_code == 0
        assert result.output.strip() == output


def test_doctor_command(monkeypatch) -> None:
    report = DoctorReport(
        supported_host=True,
        operating_system="Darwin",
        architecture="arm64",
        chip="Apple M4",
        memory_bytes=32 * 1024**3,
        python_version="3.13.0",
        metal_available=True,
        metal_error=None,
        backends=(BackendStatus("mlx", True, "0.29.0"),),
    )
    monkeypatch.setattr("mlx_one.cli.collect_doctor_report", lambda: report)

    result = CliRunner().invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "Apple M4" in result.output
    assert "MLX Metal: available" in result.output


def test_doctor_json_command(monkeypatch) -> None:
    report = DoctorReport(
        supported_host=False,
        operating_system="Linux",
        architecture="x86_64",
        chip="unknown",
        memory_bytes=None,
        python_version="3.12.0",
        metal_available=False,
        metal_error="probe failed",
        backends=(),
    )
    monkeypatch.setattr("mlx_one.cli.collect_doctor_report", lambda: report)

    result = CliRunner().invoke(main, ["doctor", "--json-output"])

    assert result.exit_code == 0
    assert '"supported_host": false' in result.output
