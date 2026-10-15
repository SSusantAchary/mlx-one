"""Local, evidence-gated model capability registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import resources
from pathlib import Path

from mlx_one.compatibility import validate_capability
from mlx_one.schemas import (
    ArtifactRef,
    CapabilitySpec,
    CapabilityStatus,
    Operation,
    RunResult,
    RunStatus,
)


class RegistryError(RuntimeError):
    """Raised when registry input is invalid or conflicts with an entry."""


class CapabilityRegistry:
    """In-memory registry keyed by exact revision, backend, operation, and constraints."""

    def __init__(self, entries: tuple[CapabilitySpec, ...] = ()) -> None:
        self._entries: dict[tuple[str, str, str, str, str], CapabilitySpec] = {}
        for entry in entries:
            self.add(entry)

    @classmethod
    def builtin(cls) -> CapabilityRegistry:
        resource = resources.files("mlx_one").joinpath("data", "capabilities.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
        return cls(tuple(CapabilitySpec.from_dict(item) for item in payload["entries"]))

    @classmethod
    def from_file(cls, path: str | Path) -> CapabilityRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise RegistryError("registry requires schema_version '1.0'")
        return cls(tuple(CapabilitySpec.from_dict(item) for item in payload["entries"]))

    def add(self, entry: CapabilitySpec) -> None:
        report = validate_capability(entry)
        if not report.valid:
            raise RegistryError("; ".join(issue.message for issue in report.issues))
        key = self._key(entry)
        if key in self._entries:
            raise RegistryError("duplicate capability entry")
        self._entries[key] = entry

    def upsert(self, entry: CapabilitySpec) -> None:
        """Insert an entry or replace the same scoped claim after validation."""
        report = validate_capability(entry)
        if not report.valid:
            raise RegistryError("; ".join(issue.message for issue in report.issues))
        self._entries[self._key(entry)] = entry

    def write(self, path: str | Path) -> Path:
        """Atomically persist a registry."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(target)
        return target

    def promote_from_result(
        self,
        entry: CapabilitySpec,
        result: RunResult,
        result_path: str | Path,
        status: CapabilityStatus | str,
    ) -> CapabilitySpec:
        """Promote a claim only from matching, successful, checksummed run evidence."""
        promoted_status = CapabilityStatus(status)
        if promoted_status not in {
            CapabilityStatus.INTEGRATION_TESTED,
            CapabilityStatus.HARDWARE_VERIFIED,
            CapabilityStatus.QUALITY_VERIFIED,
        }:
            raise RegistryError("promotion requires a verified capability status")
        if result.status is not RunStatus.COMPLETED:
            raise RegistryError("only completed runs can promote a capability")
        if result.spec.model != entry.model or result.spec.operation is not entry.operation:
            raise RegistryError("run model and operation must match the capability")
        if promoted_status in {
            CapabilityStatus.HARDWARE_VERIFIED,
            CapabilityStatus.QUALITY_VERIFIED,
        } and result.spec.hardware is None:
            raise RegistryError("hardware and quality promotion require run hardware provenance")
        source = Path(result_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        packages = result.software.packages if result.software is not None else {}
        constraints = {
            **entry.constraints,
            "evidence_run_id": result.run_id,
            "backend_version": packages.get(entry.backend, "not-recorded"),
        }
        promoted = CapabilitySpec(
            model=entry.model,
            backend=entry.backend,
            operation=entry.operation,
            status=promoted_status,
            constraints=constraints,
            hardware_profile_ids=(result.spec.hardware.profile_id,)
            if result.spec.hardware is not None
            else (),
            evidence=(
                *entry.evidence,
                ArtifactRef(
                    uri=f"run://{result.run_id}/result.json",
                    kind="run-result",
                    sha256=digest,
                    media_type="application/json",
                    metadata={"run_id": result.run_id},
                ),
            ),
            notes=entry.notes,
        )
        self._entries.pop(self._key(entry), None)
        self.upsert(promoted)
        return promoted

    def requiring_revalidation(
        self, backend_versions: dict[str, str]
    ) -> tuple[CapabilitySpec, ...]:
        """Return verified entries whose recorded backend version no longer matches."""
        verified = {
            CapabilityStatus.INTEGRATION_TESTED,
            CapabilityStatus.HARDWARE_VERIFIED,
            CapabilityStatus.QUALITY_VERIFIED,
        }
        return tuple(
            entry
            for entry in self.entries
            if entry.status in verified
            and entry.backend in backend_versions
            and entry.constraints.get("backend_version") not in {
                None,
                backend_versions[entry.backend],
            }
        )

    def find(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        backend: str | None = None,
        operation: Operation | str | None = None,
    ) -> tuple[CapabilitySpec, ...]:
        operation_value = Operation(operation).value if operation is not None else None
        return tuple(
            entry
            for entry in self._entries.values()
            if entry.model.model_id == model_id
            and (revision is None or entry.model.revision == revision)
            and (backend is None or entry.backend == backend)
            and (operation_value is None or entry.operation.value == operation_value)
        )

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "1.0", "entries": [item.to_dict() for item in self.entries]}

    @property
    def entries(self) -> tuple[CapabilitySpec, ...]:
        return tuple(self._entries.values())

    @staticmethod
    def _key(entry: CapabilitySpec) -> tuple[str, str, str, str, str]:
        constraints = json.dumps(entry.constraints, sort_keys=True, separators=(",", ":"))
        return (
            entry.model.model_id,
            entry.model.revision,
            entry.backend,
            entry.operation.value,
            constraints,
        )


# Runtime implementation status is deliberately independent of operation-level support.

class InferenceFeatureStatus(str, Enum):
    NOT_IMPLEMENTED = "not_implemented"
    EXPERIMENTAL = "experimental"
    CORRECTNESS_PASSED = "correctness_passed"
    PERFORMANCE_QUALIFIED = "performance_qualified"


@dataclass(frozen=True)
class InferenceScope:
    model_id: str
    revision: str | None
    execution_path: str
    topology_digest: str | None
    weight_precision: str | None
    cache_precision: str | None
    mlx_version: str | None
    hardware: str | None
    source_revision: str | None = None

    @property
    def complete(self) -> bool:
        import re

        return bool(
            self.revision and re.fullmatch(r"[0-9a-fA-F]{40,64}", self.revision)
            and all(asdict(self).values())
        )


@dataclass(frozen=True)
class InferenceFeatureRecord:
    feature: str
    scope: InferenceScope
    status: InferenceFeatureStatus
    evidence: tuple[ArtifactRef, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature, "scope": asdict(self.scope), "status": self.status.value,
            "evidence": [item.to_dict() for item in self.evidence],
        }


class InferenceFeatureRegistry:
    """Exact-scope native feature claims. Only validated native results can promote them."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, InferenceScope], InferenceFeatureRecord] = {}

    def find(self, feature: str, scope: InferenceScope) -> InferenceFeatureRecord | None:
        return self._records.get((feature, scope))

    def records(self) -> tuple[InferenceFeatureRecord, ...]:
        return tuple(self._records.values())

    def register_experimental(self, feature: str, scope: InferenceScope) -> None:
        self._records[(feature, scope)] = InferenceFeatureRecord(
            feature, scope, InferenceFeatureStatus.EXPERIMENTAL
        )

    def promote(
        self, feature: str, scope: InferenceScope, result_path: str | Path,
        status: InferenceFeatureStatus,
    ) -> InferenceFeatureRecord:
        status = InferenceFeatureStatus(status)
        if not scope.complete or status not in {
            InferenceFeatureStatus.CORRECTNESS_PASSED,
            InferenceFeatureStatus.PERFORMANCE_QUALIFIED,
        }:
            raise RegistryError("qualification requires complete identity and a qualified status")
        source = Path(result_path)
        raw = source.read_bytes()
        result = json.loads(raw)
        if (result.get("schema_version") != "native-inference-qualification-v4"
                or result.get("backend") != "mlx-one-native"
                or result.get("scope") != asdict(scope)
                or result.get("feature") != feature
                or result.get("status") != "passed"
                or result.get("correctness_passed") is not True):
            raise RegistryError("native qualification evidence does not match feature and scope")
        if (status is InferenceFeatureStatus.PERFORMANCE_QUALIFIED
                and result.get("performance_qualified") is not True):
            raise RegistryError("performance qualification requires measured performance gates")
        evidence = [ArtifactRef(
            uri=str(source), kind="native-inference-qualification",
            sha256=hashlib.sha256(raw).hexdigest(), media_type="application/json",
        )]
        for artifact in result.get("artifacts", []):
            path = Path(artifact["uri"])
            if not path.is_absolute():
                path = source.parent / path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact.get("sha256"):
                raise RegistryError("native qualification backing artifact checksum mismatch")
            evidence.append(ArtifactRef(uri=str(path), kind=artifact["kind"], sha256=digest,
                                        media_type=artifact.get("media_type")))
        from mlx_one.engine.adapters import ADAPTER_VERSION

        contract = result.get("execution_contract") or {}
        if (contract.get("adapter_version") != ADAPTER_VERSION
                or contract.get("topology_digest") != scope.topology_digest
                or contract.get("execution_mode") != scope.execution_path
                or not contract.get("quantization_digest") or not contract.get("components")):
            raise RegistryError("current observed native execution contract is required")
        if feature.startswith("task:"):
            from mlx_one.task_qualification import validate_task_evidence

            validate_task_evidence(result,scope,evidence,
                performance=status is InferenceFeatureStatus.PERFORMANCE_QUALIFIED)
        else:
            self._validate_native_provenance(result, scope, evidence)
        if (not feature.startswith("task:") and (
                status is InferenceFeatureStatus.PERFORMANCE_QUALIFIED
                or feature in {"quantized_paged_kv", "long_context"})):
            from mlx_one.engine.envelope import QualificationEnvelope

            try:
                QualificationEnvelope.from_mapping(result.get("workload_envelope"))
            except ValueError as exc:
                raise RegistryError("current complete workload envelope is required") from exc
        record = InferenceFeatureRecord(feature, scope, status, tuple(evidence))
        self._records[(feature, scope)] = record
        return record

    @staticmethod
    def _validate_native_provenance(result, scope, evidence):
        """Synthetic and legacy artifacts cannot stand in for checkpoint/hardware proof."""
        import re

        documents = {}
        origin = Path(evidence[0].uri).parent
        for artifact in evidence[1:]:
            name = Path(artifact.uri).name
            if name in {"measurement.json", "checkpoint.json", "manifest.json", "gate-c.json",
                        "gate-d.json", "gate-hybrid-performance.json", "gate-e.json",
                        "held-out-quality.json", "retrieval.json", "lifecycle.json",
                        "gate-p1.json"}:
                if (name not in {"held-out-quality.json", "retrieval.json"}
                        and Path(artifact.uri).parent != origin):
                    continue
                document = json.loads(Path(artifact.uri).read_text())
                if name in documents and documents[name] != document:
                    raise RegistryError("duplicate native provenance artifact")
                documents[name] = document
        measurement = documents.get("measurement.json", {})
        checkpoint = documents.get("checkpoint.json", {})
        manifest = documents.get("manifest.json", {})
        profile = manifest.get("profile", {})
        expected_precision = {"f16": "float16", "bf16": "bfloat16",
                              "q4_0": "float16,uint8", "q8_0": "float16,uint8"}.get(
                                  profile.get("cache_type"))
        files = checkpoint.get("files", {})
        native = (result.get("validation_kind") == "native-hardware"
                  and measurement.get("backend") == "mlx-metal"
                  and measurement.get("metal_available") is True
                  and checkpoint.get("revision") == scope.revision
                  and profile.get("model_id") == scope.model_id
                  and profile.get("revision") == scope.revision
                  and profile.get("inference_mode") == scope.execution_path
                  and expected_precision == scope.cache_precision
                  and bool(files) and any(n.endswith(".safetensors") for n in files)
                  and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in files.values())
                  and bool(re.fullmatch(r"[0-9a-f]{40}",
                          manifest.get("source", {}).get("git_revision", "")))
                  and bool(manifest.get("hardware", {}).get("chip"))
                  and (manifest.get("hardware", {}).get("memory_bytes") or 0) > 0)
        if not native:
            raise RegistryError("native checkpoint and Metal provenance artifacts are required")
        if (result.get("performance_qualified") is True and
                result.get("execution_contract", {}).get("quantization", {}).get("bits") == 8):
            from mlx_one.p1_qualification import P1_CHECKS, P1_GATE_VERSION

            p1 = documents.get("gate-p1.json", {})
            if (p1.get("schema_version") != P1_GATE_VERSION
                    or not P1_CHECKS <= p1.get("checks", {}).keys()
                    or p1.get("status") != "passed" or not p1.get("checks")
                    or not all(p1["checks"].values())):
                raise RegistryError("INT8 performance qualification requires measured P1 gates")
        if result.get("performance_qualified") is True:
            gates = [documents.get(name, {}) for name in (
                "gate-c.json", "gate-d.json", "gate-hybrid-performance.json", "gate-p1.json")]
            if not any(g.get("status") == "passed" and g.get("checks")
                       and all(g["checks"].values()) for g in gates):
                raise RegistryError("performance qualification requires measured gate artifacts")
        if result.get("feature") in {"quantized_paged_kv", "long_context"}:
            from mlx_one.quality_dataset import quality_coverage

            held_out = documents.get("held-out-quality.json", {})
            retrieval = documents.get("retrieval.json", {})
            context = profile.get("context_tokens", 0)
            if (result.get("quality_kind") != "native-held-out-quality-v1"
                    or result.get("quality_passed") is not True
                    or not held_out.get("dataset_identity")
                    or not quality_coverage(held_out, retrieval, context)["eligible"]):
                raise RegistryError("public KV/context qualification requires held-out quality")
        if result.get("feature") == "hybrid_prefix_restore":
            lifecycle = documents.get("lifecycle.json", {})
            gate = documents.get("gate-e.json", {})
            if (lifecycle.get("status") != "passed" or gate.get("status") != "passed"
                    or not gate.get("checks") or not all(gate["checks"].values())):
                raise RegistryError("hybrid restoration requires native lifecycle Gate E")


    def report(self, feature: str, scope: InferenceScope) -> dict[str, object]:
        record = self.find(feature, scope)
        if record is None:
            record = InferenceFeatureRecord(feature, scope, InferenceFeatureStatus.NOT_IMPLEMENTED)
        # Evidence is checked again on every use, including edits after promotion.
        for artifact in record.evidence:
            try:
                digest = hashlib.sha256(Path(artifact.uri).read_bytes()).hexdigest()
                valid = digest == artifact.sha256
            except OSError:
                valid = False
            if not valid:
                return {
                    **record.to_dict(), "status": "experimental", "stale_evidence": True,
                    "implemented": True, "checkpoint_correctness_passed": False,
                    "hardware_correctness_passed": False, "performance_qualified": False,
                    "validation_reason": "native evidence missing or checksum changed"
                }
        report = record.to_dict()
        qualified = record.status in {InferenceFeatureStatus.CORRECTNESS_PASSED,
                                      InferenceFeatureStatus.PERFORMANCE_QUALIFIED}
        return {**report, "implemented": record.status != InferenceFeatureStatus.NOT_IMPLEMENTED,
                "checkpoint_correctness_passed": qualified,
                "hardware_correctness_passed": qualified,
                "performance_qualified": (
                    record.status == InferenceFeatureStatus.PERFORMANCE_QUALIFIED),
                "validation_reason": "matching checksummed native evidence" if qualified else
                                     "native checkpoint and hardware validation pending"}
