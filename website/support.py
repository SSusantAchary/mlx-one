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
CHECKPOINT_CATALOG = "src/mlx_one/data/batch_1_sub_6b.json"
CHECKPOINT_RESULTS = "docs/qualification/batch-1-sub-6b/results.json"


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


def verified_checkpoint(root, claim, saved, record):
    """Verify a current-source profile while retaining failed diagnostic evidence."""
    identity = ("id", "model_type", "model_id", "revision", "task", "precision",
                "status", "admission_tier", "release_approval", "evidence", "numerical_status")
    if any(claim.get(key) != saved.get(key) for key in identity):
        raise ValueError("Checkpoint catalog differs from its saved result")
    if claim["status"] != "SUPPORTED" or any(
        saved.get("gates", {}).get(f"G{i}") != "PASS" for i in range(5)
    ):
        raise ValueError("Checkpoint has an incomplete functional gate")

    def receipt(reference):
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Checkpoint evidence leaves the repository")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Checkpoint evidence checksum mismatch")
        record(path)
        return json.loads(raw)

    native = receipt(claim["evidence"])
    if native.get("installed_distribution", {}).get("status") != "PASS" or any(
        native.get("interfaces", {}).get(key, {}).get("status") != "PASS"
        for key in ("cli", "engine", "http")
    ) or saved.get("interfaces", {}).get("python", {}).get("status") != "PASS":
        raise ValueError("Checkpoint installed execution or interface mismatch")
    checks = native.get("checks", {})
    failures = {key for key, value in checks.items() if value.get("status") != "PASS"}
    if native.get("status") == "PASS" and checks and not failures:
        return native
    # ViT classification was accepted with a separate checksum-bound decision.
    # Its fixed hidden-state diagnostic is never rewritten or treated as passed.
    if claim["model_type"] != "vit" or failures != {"hidden"}:
        raise ValueError("Checkpoint native execution has an unaccepted failure")
    if not all(checks.get(key, {}).get("status") == "PASS" for key in (
        "processor", "logits", "top5_parity", "public_pipeline", "fp64_rounding_diagnostic"
    )):
        raise ValueError("ViT classification checks are incomplete")
    if not claim.get("functional_decision") or claim["functional_decision"] != saved.get("functional_decision"):
        raise ValueError("ViT needs its recorded functional decision")
    decision = receipt(claim["functional_decision"])
    if (decision.get("status") != "PASS_FUNCTIONAL"
        or decision.get("fixed_1e4_hidden_diagnostic") != "FAIL"
        or decision.get("installed_interfaces_evidence") != claim["evidence"]["path"]
        or decision.get("installed_interfaces_status") != "PASS"):
        raise ValueError("ViT functional decision does not match the installed result")
    return native


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
    if len(actual) != len(set(actual)) or not set(actual) <= expected:
        raise ValueError("Release support catalog contains duplicates or unregistered model types")
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
                "availability": "0.1.1 approved release cohort",
                "quality_status": result.get("quality_status"),
                "interfaces": {
                    key: (value.get("status") if isinstance(value, dict) else value)
                    for key, value in result.get("interfaces", {}).items()
                },
            }
        )
    approved_models = list(models)
    checkpoint_catalog = json.loads((ROOT / CHECKPOINT_CATALOG).read_text())
    checkpoint_results = json.loads((ROOT / CHECKPOINT_RESULTS).read_text())
    record(ROOT / CHECKPOINT_CATALOG)
    record(ROOT / CHECKPOINT_RESULTS)
    saved = {row["id"]: row for row in checkpoint_results["models"]}
    profiles = checkpoint_catalog["models"]
    if (checkpoint_catalog["checkpoint_count"] != len(profiles)
        or len(saved) != len(profiles)
        or {row["id"] for row in profiles} != set(saved)):
        raise ValueError("Checkpoint catalog count differs from its saved results")
    for claim in profiles:
        if claim["model_type"] not in expected:
            raise ValueError("Checkpoint profile has an unregistered model type")
        native = verified_checkpoint(ROOT, claim, saved[claim["id"]], record)
        models.append({
            **claim,
            "checkpoint": claim["model_id"],
            "evidence": claim["evidence"]["path"],
            "evidence_sha256": claim["evidence"]["sha256"],
            "modality": groups[claim["model_type"]]["title"],
            "modality_id": groups[claim["model_type"]]["id"],
            "primary_operations": [claim["task"]],
            "recorded_date": checkpoint_catalog["recorded_date"],
            "availability": "Current source; next package release pending",
            "execution_dtype": "float32",
            "quality_status": native.get("quality", {}).get("status", claim["numerical_status"]),
            "interfaces": {"python": "PASS", **{
                key: value["status"] for key, value in native["interfaces"].items()
            }},
            "diagnostics": {key: value["status"] for key, value in native["checks"].items()},
        })
    identities = [(row["model_type"], row["checkpoint"], row["revision"], row["task"], row["precision"])
                  for row in models]
    if len(set(identities)) != len(models):
        raise ValueError("Duplicate supported checkpoint profile")
    supported_types = {row["model_type"] for row in models}
    for group in coverage["groups"]:
        group["supported"] = len(supported_types.intersection(group["model_types"]))
    return {
        "models": models,
        "approved_release_models": approved_models,
        "model_types": coverage["total_model_types"],
        "approved_release_model_types": len(approved_models),
        "supported": len(supported_types),
        "checkpoint_profiles": len(models),
        "checkpoint_catalog": CHECKPOINT_CATALOG,
        "modalities": len(coverage["groups"]),
        "standard": catalog["support_standard"],
        "catalog": CATALOG,
        "release": release,
        "history": [
            {
                "date": release["support_recorded_date"],
                "supported": len(approved_models),
                "source": CATALOG,
                "basis": "Dated real-checkpoint campaign records",
            },
            {
                "date": checkpoint_catalog["recorded_date"],
                "supported": len(supported_types),
                "source": CHECKPOINT_CATALOG,
                "basis": "Verified current-source checkpoints plus the dated release cohort",
            },
        ],
        "history_note": "The 8 November record covers 40 types; the 10 November record "
        "adds five types and ten checkpoint profiles. Counts are distinct architecture types. "
        "The earlier 23/7/10 audit used a different evidence policy and "
        "is not plotted as a comparable admission series.",
    }


def support_table(data):
    lines = [
        "| Model type / checkpoint | Primary operations | Precision | Status / evidence | Availability |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in data["support"]["models"]:
        lines.append(
            f"| `{row['model_type']}` — `{row['checkpoint']}` | {row['task']} | "
            f"{row['precision']} | [Supported]({row['evidence']}) | {row['availability']} |"
        )
    return "\n".join(lines)
