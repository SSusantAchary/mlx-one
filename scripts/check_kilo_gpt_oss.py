#!/usr/bin/env python3
"""Run Kilo's factorial acceptance task in an isolated disposable workspace.

Requires an already running MLX-One server and an installed Kilo CLI. Kilo executes
coding tools with its own permission policy; the inference server never executes them.
The provided workspace and all XDG state are isolated from the user's normal Kilo data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def verify_workspace(workspace, completed_tools):
    """Independently verify Kilo's generated project without changing its files."""
    report = {}
    sources = list(workspace.rglob("hello.py"))
    tests = list(workspace.rglob("test*.py"))
    assert sources, "Kilo did not create hello.py"
    assert tests, "Kilo did not create a unittest file"
    report["source_files"] = [str(path.relative_to(workspace)) for path in sources]
    report["test_files"] = [str(path.relative_to(workspace)) for path in tests]
    # Independently rerun the generated tests, without modifying Kilo's solution.
    verification = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", str(tests[0].parent), "-v"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    report["unittest"] = {
        "exit_code": verification.returncode,
        "output": verification.stdout + verification.stderr,
    }
    assert verification.returncode == 0 and "Ran 0 tests" not in report["unittest"]["output"]
    assert any("unittest" in json.dumps(tool) for tool in completed_tools), (
        "Kilo itself did not run unittest"
    )
    semantic_check = subprocess.run(
        [
            "python3",
            "-c",
            "import math, runpy, sys; "
            "module=runpy.run_path(sys.argv[1]); "
            "function=next(value for name,value in module.items() "
            "if 'factorial' in name.lower() and callable(value)); "
            "assert all(function(n)==math.factorial(n) for n in range(11))",
            str(sources[0]),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert semantic_check.returncode == 0, semantic_check.stderr
    report["independent_factorial_check"] = {"status": "PASS", "inputs": list(range(11))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kilo", default="kilo")
    parser.add_argument("--url", default="http://127.0.0.1:8084/v1")
    parser.add_argument("--chunk-timeout", type=int, default=600000)
    parser.add_argument("--model", default="gpt-oss-20b-mxfp4-128k")
    parser.add_argument("--api-key", default=os.environ.get("MLX_ONE_API_KEY"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-only", type=Path, help="Recheck an existing completed run workspace"
    )
    args = parser.parse_args()
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_only is not None:
        path = args.report_dir / "result.json"
        report = json.loads(path.read_text())
        workspace = args.verify_only.resolve()
        assert workspace == Path(report["workspace"]).resolve()
        assert report["exit_code"] == 0 and not report["client_errors"]
        assert len(report["completed_tools"]) >= 2
        report.update(verify_workspace(workspace, report["completed_tools"]))
        if report.get("failure"):
            report["previous_validation_failure"] = report.pop("failure")
        report["status"] = "PASS"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "workspace": str(workspace)}))
        return
    root = Path(
        tempfile.mkdtemp(
            prefix="mlx-kilo-", dir="/private/tmp" if Path("/private/tmp").is_dir() else None
        )
    )
    workspace = root / "workspace"
    workspace.mkdir()
    config = {
        "model": "mlx_one/" + args.model,
        "enabled_providers": ["mlx_one"],
        "share": "disabled",
        "autoupdate": False,
        "permission": {"external_directory": "deny"},
        "provider": {
            "mlx_one": {
                "name": "MLX-One Native GPT-OSS",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": args.url,
                    "apiKey": args.api_key,
                    "timeout": 1200000,
                    "chunkTimeout": args.chunk_timeout,
                },
                "models": {
                    args.model: {
                        "id": args.model,
                        "name": "Native GPT-OSS",
                        "tool_call": True,
                        "reasoning": True,
                        "temperature": True,
                        "options": {"temperature": 0, "reasoningEffort": "medium"},
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "limit": {"context": 131072, "input": 114688, "output": 16384},
                    }
                },
            }
        },
    }
    (workspace / "kilo.json").write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ)
    env["PWD"] = str(workspace)
    for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"):
        directory = root / name.lower()
        directory.mkdir()
        env[name] = str(directory)
    # Only the explicitly configured local provider is used, with no inherited inline config.
    env.pop("KILO_CONFIG_CONTENT", None)
    env["KILO_CONFIG"] = str(workspace / "kilo.json")
    env["KILO_DISABLE_AUTOUPDATE"] = "true"
    prompt = (
        "Create hello.py containing a Python function that calculates the factorial of a number. "
        "Create a small unittest file, run the tests, fix any failure, and summarize the changes. "
        "Work only inside this disposable working directory. Use python3 for running tests."
    )
    command = [
        args.kilo,
        "run",
        "--dir",
        str(workspace),
        "--pure",
        "--auto",
        "--format",
        "json",
        "--model",
        "mlx_one/" + args.model,
        prompt,
    ]
    report = {
        "workspace": str(workspace),
        "status": "FAIL",
        "model": args.model,
        "scope": "real Kilo CLI, native GPT-OSS, local file and unittest tools",
    }
    started = time.perf_counter()
    try:
        with (
            (args.report_dir / "events.jsonl").open("w") as output,
            (args.report_dir / "stderr.log").open("w") as errors,
        ):
            result = subprocess.run(
                command,
                cwd=workspace,
                env=env,
                stdout=output,
                stderr=errors,
                timeout=1200,
                check=False,
            )
        report["exit_code"] = result.returncode
        events = []
        for line in (args.report_dir / "events.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        completed_tools = []
        for event in events:
            part = event.get("part", {})
            state = part.get("state", {})
            if part.get("type") == "tool" and state.get("status") == "completed":
                completed_tools.append(
                    {
                        "tool": part.get("tool"),
                        "call_id": part.get("callID"),
                        "input": state.get("input"),
                        "output": state.get("output"),
                    }
                )
        report["completed_tools"] = completed_tools
        report["client_errors"] = [event for event in events if event.get("type") == "error"]
        assert result.returncode == 0, (
            "Kilo exited unsuccessfully; inspect stderr.log and events.jsonl"
        )
        assert not report["client_errors"], "Kilo reported protocol or generation errors"
        assert len(completed_tools) >= 2, "fewer than two completed coding tools"
        report.update(verify_workspace(workspace, completed_tools))
        report["status"] = "PASS"
    except Exception as exc:
        report["failure"] = str(exc)[:2000]
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - started
        (args.report_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in ("status", "workspace", "elapsed_s", "exit_code", "failure")
                }
            )
        )


if __name__ == "__main__":
    main()
