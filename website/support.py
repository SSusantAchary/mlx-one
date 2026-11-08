"""Verify the current support inventory without loading models or reference runtimes."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CATALOG = "src/mlx_one/data/model_support_release_20261108.json"
RELEASE = "docs/releases/0.1.1.json"


def registration_history():
    """Count actual registry snapshots; never infer admission dates from architecture dates."""
    path = "src/mlx_one/models/registry.py"
    commits = subprocess.check_output(
        ["git", "log", "--first-parent", "--reverse", "--format=%H\t%cs", "--", path],
        cwd=ROOT,
        text=True,
    )
    points = []
    for line in commits.splitlines():
        commit, day = line.split("\t")
        source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True)
        ocr = subprocess.run(
            ["git", "show", f"{commit}:src/mlx_one/ocr/registry.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        capabilities = {}
        if ocr.returncode == 0:
            for node in ast.parse(ocr.stdout).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "CAPABILITIES" for t in node.targets
                ):
                    capabilities = {
                        ast.literal_eval(key): SimpleNamespace(tasks=frozenset())
                        for key in node.value.keys
                    }
        functions = [
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {"builtin_registrations", "register_builtin_models", "ocr_registrations"}
        ]
        if not any(node.name == "builtin_registrations" for node in functions):
            original = next(node for node in functions if node.name == "register_builtin_models")
            values = next(
                node.value
                for node in original.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "registrations" for t in node.targets)
            )
            # Early registries mixed metadata and runtime wiring in one function.
            # Evaluate only its explicit metadata tuple, never runtime registration.
            original.body = [ast.Return(value=values)]

        class MetadataOnly(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None

        module = MetadataOnly().visit(ast.Module(body=functions, type_ignores=[]))
        registered = []
        namespace = {
            "ModelRegistration": lambda *args, **kwargs: args[0],
            "register_model": registered.append,
            "CAPABILITIES": capabilities,
        }
        exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
        if "builtin_registrations" in namespace:
            registered = list(namespace["builtin_registrations"]())
        else:
            registered = list(namespace["register_builtin_models"]())
        if len(registered) != len(set(registered)):
            raise ValueError(f"Duplicate registry type in historical snapshot: {commit}")
        points.append({"date": day, "commit": commit, "registered": len(registered)})
    monthly = {}
    for point in sorted(points, key=lambda p: p["date"]):
        monthly[point["date"][:7]] = point
    return {
        "points": points,
        "monthly": [dict(month=month, **point) for month, point in monthly.items()],
        "method": "First-parent native registry snapshots. Monthly values use the last dated "
        "snapshot in that month; registration is separate from functional admission.",
    }


def verified_result(root, claim, record):
    """A public support claim must match its saved checkpoint result and checksum."""
    path = (root / claim["evidence"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Support evidence leaves the repository")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != claim["evidence_sha256"]:
        raise ValueError(f"Support evidence checksum mismatch: {claim['model_type']}")
    result = json.loads(raw)
    if (
        result.get("status") != "PASS_FUNCTIONAL"
        or result.get("functional_status", "PASS") != "PASS"
        or any(result.get(key) != claim[key] for key in ("model_type", "checkpoint", "revision"))
    ):
        raise ValueError(f"Support evidence identity or execution mismatch: {claim['model_type']}")
    record(path)
    return result


def current_support(record, coverage):
    catalog_path = ROOT / CATALOG
    release_path = ROOT / RELEASE
    catalog = json.loads(catalog_path.read_text())
    release = json.loads(release_path.read_text())
    record(catalog_path)
    record(release_path)
    record(Path(__file__))
    if (
        release["support_catalog"] != CATALOG
        or release["support_catalog_sha256"]
        != hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Release decision does not match the support catalog")
    expected = {name for group in coverage["groups"] for name in group["model_types"]}
    actual = [row["model_type"] for row in catalog["models"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Support catalog must cover the registered model types exactly once")
    if catalog["model_type_count"] != len(actual):
        raise ValueError("Support catalog count differs from its records")
    groups = {name: group for group in coverage["groups"] for name in group["model_types"]}
    models = []
    for claim in catalog["models"]:
        if claim["status"] != "SUPPORTED":
            raise ValueError("Current supported catalog contains an unadmitted model")
        result = verified_result(ROOT, claim, record)
        models.append(
            {
                **claim,
                "modality": groups[claim["model_type"]]["title"],
                "modality_id": groups[claim["model_type"]]["id"],
                "task": ", ".join(claim["primary_operations"]),
                "recorded_date": release["support_recorded_date"],
                "quality_status": result.get("quality_status"),
                "interfaces": {
                    key: (value.get("status") if isinstance(value, dict) else value)
                    for key, value in result.get("interfaces", {}).items()
                },
            }
        )
    for group in coverage["groups"]:
        group["supported"] = sum(row["model_type"] in group["model_types"] for row in models)
    return {
        "models": models,
        "model_types": len(models),
        "supported": len(models),
        "modalities": len(coverage["groups"]),
        "standard": catalog["support_standard"],
        "catalog": CATALOG,
        "release": release,
        "history": [
            {
                "date": release["support_recorded_date"],
                "supported": len(models),
                "source": CATALOG,
                "basis": "Dated real-checkpoint campaign records",
            }
        ],
        "history_note": "The 8 November campaign records 40 selected primary workflows. "
        "The earlier 23/7/10 audit used a different evidence policy and "
        "is not plotted as a comparable admission series.",
    }


def support_table(data):
    lines = [
        "| Model type / checkpoint | Primary operations | Precision | Status / evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in data["support"]["models"]:
        lines.append(
            f"| `{row['model_type']}` — `{row['checkpoint']}` | {row['task']} | "
            f"{row['precision']} | [Supported]({row['evidence']}) |"
        )
    return "\n".join(lines)
