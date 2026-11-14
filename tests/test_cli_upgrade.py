"""Git-only upgrades must target the invoking environment without shell execution."""

import subprocess
import sys

import pytest
from click.testing import CliRunner

from mlx_one.cli import main
from mlx_one.cli_upgrade import GIT_REPOSITORY


def test_git_upgrade_uses_current_python_and_reinstalls_same_version(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mlx_one.cli_upgrade.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    result = CliRunner().invoke(main, ["upgrade"])
    assert result.exit_code == 0, result.output
    command, settings = calls[0]
    assert command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        f"mlx-one[server] @ {GIT_REPOSITORY}@main",
    ]
    assert settings == {"check": True}


def test_git_upgrade_dry_run_never_installs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("A dry run must not invoke pip")

    monkeypatch.setattr("mlx_one.cli_upgrade.subprocess.run", forbidden)
    result = CliRunner().invoke(
        main,
        [
            "upgrade",
            "--ref",
            "feature/gptoss",
            "--extras",
            "server,vision,server",
            "--no-deps",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert f"mlx-one[server,vision] @ {GIT_REPOSITORY}@feature/gptoss" in result.output
    assert "--no-deps" in result.output


def test_core_only_and_pinned_commit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mlx_one.cli_upgrade.subprocess.run", lambda command, **_kwargs: calls.append(command)
    )
    revision = "f" * 40
    result = CliRunner().invoke(main, ["upgrade", "--ref", revision, "--extras", ""])
    assert result.exit_code == 0
    assert calls[0][-1] == f"mlx-one @ {GIT_REPOSITORY}@{revision}"


@pytest.mark.parametrize(
    "options",
    [
        ["--ref", "main;echo bad"],
        ["--ref", "--bad"],
        ["--ref", "main#other"],
        ["--ref", "../main"],
        ["--extras", "server] @ https://other.invalid"],
    ],
)
def test_invalid_install_targets_do_not_run_pip(monkeypatch, options):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid input must fail before pip")

    monkeypatch.setattr("mlx_one.cli_upgrade.subprocess.run", forbidden)
    result = CliRunner().invoke(main, ["upgrade", *options])
    assert result.exit_code == 2


def test_upgrade_reports_install_failure(monkeypatch):
    def failed(command, **_kwargs):
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr("mlx_one.cli_upgrade.subprocess.run", failed)
    result = CliRunner().invoke(main, ["upgrade"])
    assert result.exit_code == 1
    assert "Git upgrade failed (exit 7)" in result.output
