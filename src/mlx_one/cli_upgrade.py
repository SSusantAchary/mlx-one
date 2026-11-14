"""Explicit Git upgrades using the Python environment that owns this CLI."""

import re
import shlex
import subprocess
import sys

import click

GIT_REPOSITORY = "git+https://github.com/SSusantAchary/mlx-one.git"


def git_ref(_ctx, _param, value):
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
        or ".." in value
        or "//" in value
        or value.endswith(("/", ".lock"))
    ):
        raise click.BadParameter("use a Git branch, tag or commit hash")
    return value


def git_extras(_ctx, _param, value):
    extras = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in extras):
        raise click.BadParameter("use comma-separated extra names, e.g. server,vision,ocr")
    return extras


@click.command("upgrade", help="Upgrade mlx-one from Git in this CLI's Python environment.")
@click.option(
    "--ref",
    default="main",
    show_default=True,
    callback=git_ref,
    help="Git branch, tag or commit to install.",
)
@click.option(
    "--extras",
    default="server",
    show_default=True,
    callback=git_extras,
    help="Comma-separated extras; use an empty string for core only.",
)
@click.option(
    "--no-deps", is_flag=True, help="Reinstall only mlx-one; keep dependencies unchanged."
)
@click.option("--dry-run", is_flag=True, help="Print the command without installing or networking.")
def upgrade_command(ref, extras, no_deps, dry_run):
    package = "mlx-one" + ("[" + ",".join(extras) + "]" if extras else "")
    requirement = f"{package} @ {GIT_REPOSITORY}@{ref}"
    # Force the Git code to be replaced even when its version label is unchanged.
    # No shell, sudo, alternate interpreter or package-index fallback for mlx-one.
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall"]
    if no_deps:
        command.append("--no-deps")
    command.append(requirement)
    click.echo(f"Python environment: {sys.prefix}")
    click.echo(shlex.join(command))
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"Git upgrade failed (exit {exc.returncode}).") from exc
    except OSError as exc:
        raise click.ClickException(f"Could not run pip: {exc}") from exc
    click.echo("Git installation completed. Run mlx-one --version in a new process.")
