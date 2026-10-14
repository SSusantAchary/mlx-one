"""PR-00 registry inventory and isolated native prefix diagnostics, never promotion."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.inference_baseline import source_provenance
from mlx_one.inference_qualification import write_integrity
from mlx_one.p1_profiles import CHECKPOINTS

BASE_SHA = "85d522ca0aa428c006ae939465d14301a9b0a4c2"
INVENTORY_VERSION = "native-runtime-baseline-v1"


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_inventory(registrations=None):
    """Describe declarations without resolving loaders, adapters or native math."""
    registrations = (list(registrations) if registrations is not None else
                     [get_registration(name) for name in registered_model_types()])
    rows = []
    for entry in sorted(registrations, key=lambda r: r.model_type):
        from mlx_one.engine.adapter_registry import get_runtime_registration

        try:
            binding = (get_runtime_registration(entry.model_type)
                       if get_registration(entry.model_type) == entry else None)
        except KeyError:
            binding = None
        component = entry.model_type == "sam3_tracker_video"
        kind = ("component_only" if component else "architecture_only" if not entry.loader_path
                and not entry.tasks else "task_unavailable" if not entry.loader_path
                else "reference_loader_declared")
        tasks = {task: "loader_declared" if entry.loader_path else "unavailable"
                 for task in sorted(entry.tasks)}
        if entry.model_type == "qwen3_5":
            tasks["vision-language-generation"] = "unavailable"
        reason = ("Nested native video tracker; no standalone loader" if component else
                  "No registered task loader" if not entry.loader_path else
                  "Declared loader; shared-service integration and qualification not established")
        rows.append({"model_type": entry.model_type, "modality": entry.modality,
            "config_path": entry.config_path, "model_path": entry.model_path,
            "loader_path": entry.loader_path, "sanitizer_path": entry.sanitizer_path,
            "weight_contract_path": entry.weight_contract_path,
            "declared_capabilities": sorted(entry.capabilities),
            "declared_tasks": sorted(entry.tasks),
            "integration_kind": kind, "reason": reason, "task_status": tasks,
            "parent_model_type": "sam3_video" if component else None,
            "architecture_status": "registered", "shared_service_integrated": bool(binding
                and binding.integration_status == "integrated"),
            "runtime_contract_version": binding.contract_version if binding else None,
            "correctness_validation": "not_run", "performance_qualification": "not_run",
            "pinned_recipe_keys": [k for k, c in CHECKPOINTS.items()
                                   if c.architecture == entry.model_type]})
    return {"registration_count": len(rows), "registrations": rows,
            "qualification_scope": "this inventory run; historical evidence is not promoted",
            "qualification_promoted": False}


def checkpoint_inventory():
    """Checksum local assets for existing pinned recipes; never implicitly download."""
    from huggingface_hub import snapshot_download

    rows = {}
    for key, checkpoint in CHECKPOINTS.items():
        row = {"model_id": checkpoint.model_id, "revision": checkpoint.revision,
               "conversion_required": checkpoint.conversion_required,
               "execution_validated": False, "qualification": "not_run"}
        try:
            root = Path(snapshot_download(checkpoint.model_id, revision=checkpoint.revision,
                                          local_files_only=True))
            files = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*"))
                     if p.is_file()}
            tokenizers = ("tokenizer.json", "tokenizer.model", "vocab.json")
            complete = ("config.json" in files and any(p.endswith(".safetensors") for p in files)
                        and any(p in files for p in tokenizers))
            rows[key] = {**row, "status": "local" if complete else "incomplete",
                         "path": str(root.resolve()), "files": files,
                         "reason": None if complete else "missing config, tokenizer or safetensors"}
        except (OSError, ValueError) as exc:
            rows[key] = {**row, "status": "missing", "reason": str(exc)}
    return rows


def effective_ceiling(preflight_ceiling, configured=None):
    if configured is not None and (isinstance(configured, bool) or configured < 1):
        raise ValueError("configured process ceiling must be positive")
    return min(preflight_ceiling, configured) if configured is not None else preflight_ceiling


def diagnostic_outcome(supervision, result):
    completed = supervision.get("status") == "completed" and result is not None
    reproduced = bool(completed and result.get("status") == "failed"
                      and result.get("failure_reproduced")
                      and result.get("cleanup_passed"))
    passed = bool(completed and result.get("status") == "passed" and result.get("cleanup_passed"))
    qualification = result.get("status", "not_run") if completed else "not_run"
    if qualification == "passed" and not passed:
        qualification = "failed"
    return {"status": "COMPLETED" if completed else "FAIL",
            "readiness": "parity_passed" if passed else "reproduced" if reproduced
                else "not_reproduced",
            "qualification": qualification,
            "qualification_promoted": False, "worker_status": supervision.get("status")}


def run_baseline(output, *, native=False, configured_ceiling=None, worker_output=None,
                 require_parity=False):
    from mlx_one.engine.adapter_registry import runtime_coverage
    from mlx_one.engine.pressure import _command, sample_telemetry
    from mlx_one.native_qualification import preflight, supervised_command, write

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    dependencies = {"python": platform.python_version()}
    for name in ("mlx", "numpy", "tokenizers", "huggingface-hub", "safetensors", "fastapi",
                 "uvicorn", "pillow", "pytest", "ruff"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    assets = checkpoint_inventory()
    source = source_provenance(root)
    tests = {str(p.relative_to(root)): file_digest(p)
             for p in sorted((root / "tests").glob("*.py"))}
    telemetry = asdict(sample_telemetry())
    snapshot = {"schema_version": INVENTORY_VERSION, "base_sha": BASE_SHA, "source": source,
        "dependencies": dependencies, "test_files": tests, "registry": registry_inventory(),
        "runtime_contracts": runtime_coverage(),
        "hardware": {"platform": platform.platform(), "architecture": platform.machine(),
            "chip": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "telemetry": telemetry}, "native_preflight": "not_run",
        "telemetry_limitations": [key for key, value in telemetry.items() if value is None],
        "asset_blockers": {key: row.get("reason") for key, row in assets.items()
                           if row.get("status") != "local"},
        "asset_scope": "six existing P1 recipes; other task assets not assessed",
        "qualification_promoted": False}
    write(output / "inventory.json", snapshot)
    write(output / "assets.json", assets)
    manifest = {"schema_version": INVENTORY_VERSION, "source": source, "base_sha": BASE_SHA,
        "checkpoint": "qwen3-small", "asset": assets.get("qwen3-small"),
        "configuration": {"execution_dtype": "float16", "key_type": "f16", "value_type": "f16",
            "mode": "experimental_dense_batch", "forced_steps": 32,
            "require_fresh_full_prefill_parity": require_parity,
            "configured_process_ceiling_bytes": configured_ceiling},
        "qualification_promoted": False}
    write(output / "manifest.json", manifest)
    write_integrity(output)
    if not native:
        return 0
    if assets.get("qwen3-small", {}).get("status") != "local":
        write(output / "diagnostic.json", {"status": "NOT_RUN", "qualification": "not_run",
            "reason": "pinned qwen3-small assets missing or incomplete",
            "qualification_promoted": False})
        write_integrity(output)
        return 1
    try:
        observed = preflight(output / "preflight", configured_ceiling=configured_ceiling)
        ceiling = effective_ceiling(observed["process_ceiling_bytes"], configured_ceiling)
    except (OSError, ValueError, RuntimeError) as exc:
        write(output / "diagnostic.json", {"status": "NOT_RUN", "qualification": "not_run",
            "reason": str(exc), "qualification_promoted": False})
        write_integrity(output)
        return 1
    snapshot["native_preflight"] = "passed"
    snapshot["native_preflight_artifact"] = str((output / "preflight/preflight.json").resolve())
    write(output / "inventory.json", snapshot)
    destination = Path(worker_output) if worker_output else output / "prefix-diagnostic"
    manifest.update(output=str(destination.resolve()), process_ceiling_bytes=ceiling)
    write(output / "manifest.json", manifest)
    command = [sys.executable, "-m", "mlx_one.runtime_baseline",
               str((output / "manifest.json").resolve())]
    supervision = supervised_command(command, destination, ceiling=ceiling, timeout=600,
        env={"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    result_path = destination / "prefix-diagnostic.json"
    try:
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
    except (OSError, ValueError):
        result = None
    outcome = diagnostic_outcome(supervision, result)
    outcome.update(artifact=str(destination.resolve()), process_ceiling_bytes=ceiling)
    write(output / "diagnostic.json", outcome)
    write_integrity(output)
    accepted = {"parity_passed"} if require_parity else {"reproduced", "parity_passed"}
    return 0 if outcome["readiness"] in accepted else 1


def validate_prefix_manifest(manifest):
    """Reject substituted inputs before importing model execution."""
    root = Path(__file__).resolve().parents[2]
    if source_provenance(root)["source_sha256"] != manifest["source"]["source_sha256"]:
        raise ValueError("source changed before native prefix reproduction")
    checkpoint = CHECKPOINTS["qwen3-small"]
    asset = manifest["asset"]
    expected = {"execution_dtype": "float16", "key_type": "f16", "value_type": "f16",
                "mode": "experimental_dense_batch", "forced_steps": 32}
    if any(manifest["configuration"].get(key) != value for key, value in expected.items()):
        raise ValueError("prefix diagnostic execution or precision substitution")
    if (asset["model_id"], asset["revision"]) != (checkpoint.model_id, checkpoint.revision):
        raise ValueError("prefix diagnostic checkpoint identity mismatch")
    for name, digest in asset["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid checkpoint asset path")
        if file_digest(Path(asset["path"]) / name) != digest:
            raise ValueError("checkpoint assets changed before native execution")


def prefix_worker(manifest):
    validate_prefix_manifest(manifest)
    import mlx.core as mx

    from mlx_one.engine.adapters import native_adapter, quantized_modules
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.parity import prefix_failure_diagnostic
    from mlx_one.engine.reporting import tensor_dtypes
    from mlx_one.native_qualification import write
    from mlx_one.text import load_text_model

    output = Path(manifest["output"])
    asset = manifest["asset"]
    checkpoint = CHECKPOINTS["qwen3-small"]
    mx.set_default_device(mx.gpu)
    if not mx.metal.is_available():
        raise RuntimeError("native diagnostic requires Metal")
    bundle = load_text_model(checkpoint.model_id, revision=checkpoint.revision, offline=True)
    if (bundle.path.resolve() != Path(asset["path"]).resolve()
            or bundle.quantization.get("bits") != 8 or bundle.quantization.get("group_size") != 64):
        raise ValueError("prefix diagnostic checkpoint or weight precision substitution")
    configure_execution_dtype(bundle, "float16")
    adapter = native_adapter(bundle)
    tensors = {}
    def capture(name, logits):
        # Compact owned arrays avoid retaining a full prefill logits allocation.
        copy = mx.zeros(logits.shape, dtype=logits.dtype)
        copy[...] = logits
        mx.eval(copy)
        tensors[name] = copy
    result = prefix_failure_diagnostic(bundle, sink=capture)
    payload = output / "logits.safetensors"
    mx.save_safetensors(str(payload), tensors)
    result.update(source=manifest["source"], checkpoint=asset,
        observed_weight_dtypes=tensor_dtypes(bundle.model.parameters()),
        observed_quantized_modules=[asdict(spec) for spec in quantized_modules(bundle)],
        quantization_digest=adapter.spec.quantization_digest,
        execution_policy=bundle.model._mlx_one_execution_policy,
        native_execution={"device": str(mx.default_device()), "metal_available": True,
            "mlx_active_bytes": mx.get_active_memory(), "mlx_cached_bytes": mx.get_cache_memory(),
            "mlx_peak_bytes": mx.get_peak_memory()},
        tensor_artifacts=[{"uri": str(payload.resolve()), "sha256": file_digest(payload),
                           "kind": "native-prefix-logits"}])
    write(output / "prefix-diagnostic.json", result)
    write_integrity(output)


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        prefix_worker(request)
    except Exception as exc:
        from mlx_one.native_qualification import write

        write(Path(request["output"]) / "failure.json", {"status": "failed",
              "reason": f"{type(exc).__name__}: {exc}", "qualification_promoted": False})
        write_integrity(request["output"])
        raise
