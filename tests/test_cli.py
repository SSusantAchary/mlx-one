from click.testing import CliRunner

from mlx_one import __version__
from mlx_one.cli import main
from mlx_one.diagnostics import BackendStatus, DoctorReport
from mlx_one.schemas import InspectionResult, Modality, ModelSource, ModelSpec


def test_version() -> None:
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    for command in (
        "doctor",
        "inspect",
        "hardware",
        "plan",
        "calibrate",
        "benchmark",
        "eval",
        "compare",
        "merge",
        "ship",
    ):
        assert command in result.output
    for implemented_command in (
        "benchmark",
        "data",
        "export",
        "evaluate",
        "registry",
        "train",
    ):
        assert f"\n  {implemented_command} " in result.output
    for deferred_command in ("pipeline",):
        assert f"\n  {deferred_command} " not in result.output


def test_eval_requires_evaluation_inputs() -> None:
    result = CliRunner().invoke(main, ["eval"])

    assert result.exit_code == 2
    assert "--model, --revision, and --dataset are required" in result.output


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


def inspection_result() -> InspectionResult:
    return InspectionResult(
        model=ModelSpec(
            model_id="organization/model",
            revision="a" * 40,
            modality=Modality.TEXT,
            architectures=("ExampleForCausalLM",),
            model_type="example",
            parameter_count=1_000_000,
            parameter_count_source="hub-safetensors",
            weight_format="safetensors",
            weight_bytes=2_000_000,
            dtype="BF16",
            license="mit",
        ),
        source=ModelSource.HUGGING_FACE,
        inspected_at="2026-09-06T12:00:00Z",
    )


def test_inspect_human_output(monkeypatch) -> None:
    monkeypatch.setattr("mlx_one.cli.inspect_model", lambda *_args, **_kwargs: inspection_result())

    result = CliRunner().invoke(main, ["inspect", "organization/model", "--revision", "main"])

    assert result.exit_code == 0
    assert "Model: organization/model" in result.output
    assert "Parameters: 1.00M (hub-safetensors)" in result.output


def test_inspect_json_and_file_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("mlx_one.cli.inspect_model", lambda *_args, **_kwargs: inspection_result())
    output = tmp_path / "inspection.json"

    result = CliRunner().invoke(
        main,
        ["inspect", "organization/model", "--json-output", "--output", str(output)],
    )

    assert result.exit_code == 0
    assert '"source": "hugging-face"' in result.output
    assert output.exists()
    assert "Saved JSON" not in result.output


def test_inspect_maps_operational_error(monkeypatch) -> None:
    from mlx_one.inspection import InspectionError

    def fail(*_args, **_kwargs):
        raise InspectionError("model metadata is unavailable")

    monkeypatch.setattr("mlx_one.cli.inspect_model", fail)

    result = CliRunner().invoke(main, ["inspect", "organization/model"])

    assert result.exit_code == 1
    assert "Error: model metadata is unavailable" in result.output
    assert "Traceback" not in result.output


def test_hardware_commands() -> None:
    runner = CliRunner()

    listed = runner.invoke(main, ["hardware", "list"])
    shown = runner.invoke(main, ["hardware", "show", "m4-air-32gb", "--json-output"])

    assert listed.exit_code == 0
    assert "m4-air-32gb" in listed.output
    assert shown.exit_code == 0
    assert '"process_memory_budget_bytes": 25769803776' in shown.output


def test_plan_inference_json_and_atomic_output(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        """{
          "model_type": "qwen2",
          "architectures": ["Qwen2ForCausalLM"],
          "num_parameters": 500000000,
          "hidden_size": 896,
          "num_hidden_layers": 24,
          "num_attention_heads": 14,
          "num_key_value_heads": 2
        }""",
        encoding="utf-8",
    )
    output = tmp_path / "plan.json"

    result = CliRunner().invoke(
        main,
        [
            "plan",
            "inference",
            "--model",
            str(model),
            "--hardware",
            "m4-air-32gb",
            "--json-output",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"fit": "comfortable"' in result.output
    assert output.exists()
    assert "Saved JSON" not in result.output


def test_plan_train_auto_human_output(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen2","num_parameters":1500000000}', encoding="utf-8"
    )

    result = CliRunner().invoke(
        main,
        [
            "plan",
            "train",
            "--model",
            str(model),
            "--hardware",
            "m4-air-32gb",
            "--method",
            "auto",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Auto-selected method:" in result.output
    assert "not compatibility guarantees" in result.output


def test_calibration_dry_run_cli(tmp_path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "calibrate",
            "text",
            "--matrix",
            "m4-air-32gb-text-v1",
            "--output",
            str(tmp_path / "results"),
            "--dry-run",
            "--select",
            "0.5b-inference-bf16-c512$",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Cells: 1" in result.output
    assert not (tmp_path / "results").exists()
