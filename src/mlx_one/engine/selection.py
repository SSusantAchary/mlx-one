"""Opt-in, exact-evidence selection. No performance claims are inferred."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from mlx_one.engine.modes import validate_mode
from mlx_one.engine.reporting import native_report
from mlx_one.registry import InferenceFeatureStatus


def load_evidence(paths):
    from mlx_one.registry import InferenceFeatureRegistry, InferenceScope

    registry = InferenceFeatureRegistry()
    for path in paths:
        proof = json.loads(Path(path).read_text())
        status = (InferenceFeatureStatus.PERFORMANCE_QUALIFIED if proof.get("performance_qualified")
                  else InferenceFeatureStatus.CORRECTNESS_PASSED)
        registry.promote(proof["feature"], InferenceScope(**proof["scope"]), path, status)
    return registry


def host_identity():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=1, check=False).stdout.strip()
        ram = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=1, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        chip = ram = ""
    return json.dumps({"chip": chip or platform.machine(), "ram": ram or None,
                       "os": platform.platform()}, sort_keys=True)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return "source-sha256:" + digest.hexdigest()


def compatibility_digest(runtime):
    fingerprint = asdict(runtime.fingerprint(namespace="qualification",
                                             key_type=runtime.config.key_type,
                                             value_type=runtime.config.value_type))
    fingerprint.pop("namespace")
    fingerprint["model_identity"] = runtime.weight_identity
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()


def select_auto_mode(bundle, runtime, registry, profile, *, hardware=None, source=None):
    fallback = {"requested_mode": "auto", "resolved_mode": "reference_dense",
                "reason": "no matching performance-qualified execution", "evidence_scope": None}
    if registry is None or not profile:
        return {**fallback, "reason": "qualification registry and workload envelope are required"}
    required = {"context_tokens", "slots", "chunk_tokens", "ttft_budget_ms", "decode_gap_budget_ms",
                "output_tokens", "page_tokens", "cache_type", "admission_policy"}
    if not required <= profile.keys() or any(profile[k] is None for k in required):
        return {**fallback, "reason": "workload envelope or service budgets are incomplete"}
    hardware, source = hardware or host_identity(), source or source_identity()
    precision = {"f16": "float16", "bf16": "bfloat16", "q4_0": "float16,uint8",
                 "q8_0": "float16,uint8"}.get(runtime.config.key_type)
    if precision is None or runtime.config.key_type != runtime.config.value_type:
        return {**fallback, "reason": "auto cannot select unqualified cache precision"}
    if (profile["cache_type"] != runtime.config.key_type
            or profile["page_tokens"] != runtime.config.block_size_tokens):
        return {**fallback, "reason": "configured cache layout does not match workload envelope"}
    candidates = {"paged": "paged_attention", "experimental_hybrid_batch": "hybrid_prefix_restore",
                  "experimental_dense_batch": "continuous_batching"}
    if runtime.config.key_type in {"q4_0", "q8_0"}:
        candidates = {"paged": "quantized_paged_kv"}
    for mode, feature in candidates.items():
        try:
            validate_mode(bundle, runtime.plan, mode)
        except ValueError:
            continue
        scope = native_report(bundle, runtime, mode=mode, hardware=hardware,
                              source_revision=source)["scope"]
        scope["cache_precision"] = precision
        from mlx_one.registry import InferenceScope

        expected = InferenceScope(**scope)
        for record in registry.records():
            if record.feature != feature or record.scope != expected:
                continue
            fresh = registry.report(feature, record.scope)
            if fresh["status"] != InferenceFeatureStatus.PERFORMANCE_QUALIFIED.value:
                continue
            proof = json.loads(Path(record.evidence[0].uri).read_text())
            if (proof.get("workload_envelope") != profile
                    or proof.get("compatibility_digest") != compatibility_digest(runtime)):
                continue
            from mlx_one.engine.paged import KERNEL_VERSION

            if mode == "paged" and proof.get("kernel_version") != KERNEL_VERSION:
                continue
            if feature == "quantized_paged_kv" and (
                    proof.get("quality_passed") is not True
                    or proof.get("kernel_correctness_passed") is not True
                    or proof.get("quality_kind") != "native-held-out-quality-v1"
                    or proof.get("quality_coverage_passed") is not True):
                continue
            return {"requested_mode": "auto", "resolved_mode": mode,
                    "reason": "matching checksummed performance qualification",
                    "evidence_scope": asdict(record.scope)}
    return fallback


def qualification_matches(runtime, feature, *, profile=None, performance=False):
    registry = getattr(runtime, "qualification_registry", None)
    if registry is None:
        return False
    profile = profile or getattr(runtime, "qualification_profile", None)
    if not profile:
        return False
    batching = getattr(runtime, "batching_config", None)
    if batching is None:
        return False
    try:
        validate_execution_envelope(runtime, batching, required=True)
    except ValueError:
        return False
    mode = getattr(runtime, "mode", None)
    expected = native_report(runtime.bundle, runtime, mode=mode, hardware=host_identity(),
                             source_revision=source_identity())["scope"]
    expected["cache_precision"] = {"f16": "float16", "bf16": "bfloat16",
                                   "q4_0": "float16,uint8", "q8_0": "float16,uint8"}.get(
                                       runtime.config.key_type)
    for record in registry.records():
        if record.feature != feature or asdict(record.scope) != expected:
            continue
        status = registry.report(feature, record.scope)["status"]
        accepted = ({"performance_qualified"} if performance else
                    {"correctness_passed", "performance_qualified"})
        if status not in accepted:
            continue
        proof = json.loads(Path(record.evidence[0].uri).read_text())
        if proof.get("compatibility_digest") != compatibility_digest(runtime):
            continue
        if proof.get("workload_envelope") != profile:
            continue
        from mlx_one.engine.paged import KERNEL_VERSION

        if mode == "paged" and proof.get("kernel_version") != KERNEL_VERSION:
            continue
        if (proof.get("quality_passed") is not True or (feature == "long_context"
                and (proof.get("memory_passed") is not True
                     or proof.get("service_passed") is not True))):
            continue
        if feature == "quantized_paged_kv":
            from mlx_one.quality_dataset import QUALITY_SCHEMA

            if (proof.get("kernel_correctness_passed") is not True
                    or proof.get("quality_kind") != QUALITY_SCHEMA
                    or proof.get("quality_coverage_passed") is not True):
                continue
        return True
    return False


def validate_execution_envelope(runtime, batching, *, required=False):
    """Bind an immutable qualification envelope to the configured execution settings."""
    profile = getattr(runtime, "qualification_profile", None)
    if not profile:
        if required:
            from mlx_one.engine.modes import ModelCapabilityUnsupported

            raise ModelCapabilityUnsupported("qualified execution requires a workload envelope")
        return
    settings = {"slots": batching.max_sequences, "chunk_tokens": batching.prefill_chunk_size,
                "page_tokens": runtime.config.block_size_tokens,
                "cache_type": runtime.config.key_type,
                "admission_policy": batching.admission_policy}
    if any(profile.get(k) != value for k, value in settings.items()):
        from mlx_one.engine.modes import ModelCapabilityUnsupported

        raise ModelCapabilityUnsupported(
            "configured execution does not match qualification envelope")
    runtime.qualified_envelope = dict(profile)


def validate_context_admission(runtime, prompt_tokens, output_tokens):
    """Public long-context admission is evidence gated; harnesses still preflight topology."""
    total = prompt_tokens + output_tokens
    envelope = getattr(runtime, "qualified_envelope", None)
    if envelope is not None and (prompt_tokens > envelope["context_tokens"]
                                 or output_tokens > envelope["output_tokens"]):
        from mlx_one.engine.modes import ModelCapabilityUnsupported

        raise ModelCapabilityUnsupported("request exceeds the qualified workload envelope")
    if total > runtime.plan.max_context_tokens:
        from mlx_one.engine.modes import ModelCapabilityUnsupported

        raise ModelCapabilityUnsupported("prompt plus output exceeds checkpoint context")
    if total >= 128000 and not runtime.harness_only and not qualification_matches(
            runtime, "long_context", profile=getattr(runtime, "qualification_profile", None)):
        from mlx_one.engine.modes import ModelCapabilityUnsupported

        raise ModelCapabilityUnsupported("128K-class admission requires matching qualification")
