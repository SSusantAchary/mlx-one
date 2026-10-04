"""Check guide coverage, repository links, Python syntax and audio API examples.

No model imports, network access, downloads or Metal execution are required.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import shlex
from pathlib import Path
from urllib.parse import unquote, urlsplit

from mlx_one import audio
from mlx_one.cli import main as cli
from mlx_one.core.registry import get_registration, registered_model_types

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path = ROOT) -> dict:
    readme = root / "README.md"
    index = readme.read_text()
    files = sorted((root / "src/mlx_one/models").glob("*/*/README.md"))
    files += [readme, root / "docs/model-catalog.md", root / "docs/audio-qualification-m4-32gb.md"]
    errors, coverage = [], {}
    for kind in registered_model_types():
        registration = get_registration(kind)
        module = registration.config_path.partition(":")[0]
        guide = root / "src" / Path(*module.split(".")).parent / "README.md"
        relative = guide.relative_to(root).as_posix()
        coverage[kind] = relative
        if not guide.is_file() or f"]({relative})" not in index:
            errors.append(f"{kind}: missing linked implementation guide {relative}")
    blocks = 0
    for path in files:
        text = path.read_text()
        # Links in code fences are input examples, not navigational links.
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", prose):
            target = target.strip().split()[0].strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            resolved = (path.parent / unquote(parts.path)).resolve()
            if not resolved.exists():
                errors.append(f"{path.relative_to(root)}: missing local link {target}")
        for language, code in re.findall(r"```(\w+)\n(.*?)```", text, flags=re.S):
            if language in {"python", "py"}:
                blocks += 1
                try:
                    tree = ast.parse(code)
                except SyntaxError as exc:
                    errors.append(f"{path.relative_to(root)}: Python example {exc}")
                    continue
                # Validate literal imports and call signatures in audio guides.
                bindings = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "mlx_one.audio":
                        for name in node.names:
                            value = getattr(audio, name.name, None)
                            if value is None:
                                errors.append(
                                    f"{path.relative_to(root)}: missing audio export {name.name}"
                                )
                            else:
                                bindings[name.asname or name.name] = value
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                        continue
                    value = bindings.get(node.func.id)
                    if value is None or any(isinstance(arg, ast.Starred) for arg in node.args):
                        continue
                    if any(keyword.arg is None for keyword in node.keywords):
                        continue
                    try:
                        inspect.signature(value).bind(
                            *([None] * len(node.args)),
                            **{keyword.arg: None for keyword in node.keywords},
                        )
                    except (TypeError, ValueError) as exc:
                        errors.append(f"{path.relative_to(root)}: {node.func.id} signature: {exc}")
            if language in {"bash", "sh"}:
                for line in code.replace("\\\n", " ").splitlines():
                    if not line.strip().startswith("mlx-one "):
                        continue
                    tokens = shlex.split(line, comments=True)
                    if len(tokens) < 2 or tokens[1] not in {"transcribe", "align"}:
                        continue
                    command = cli.commands[tokens[1]]
                    flags = {
                        flag
                        for param in command.params
                        for flag in (*param.opts, *getattr(param, "secondary_opts", ()))
                    }
                    for token in tokens[2:]:
                        if token.startswith("--") and token.split("=", 1)[0] not in flags:
                            errors.append(
                                f"{path.relative_to(root)}: unsupported CLI option {token}"
                            )
    return {
        "status": "FAIL" if errors else "PASS",
        "registered_model_types": len(coverage),
        "guide_files": len(set(coverage.values())),
        "python_examples": blocks,
        "coverage": coverage,
        "errors": errors,
    }


if __name__ == "__main__":
    report = check()
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["errors"] else 0)
