"""Per-request opt-in selection; the disabled path does not probe a backend."""

from __future__ import annotations

import logging

from .capabilities import native_status
from .registry import Registry
from .schemas import DispatchDecision, NativeConfig, NativeKernelUnavailable

logger = logging.getLogger(__name__)


def choose(request, config=None, registry=None, *, status_provider=native_status):
    config = NativeConfig.from_env() if config is None else config

    def fallback(reason):
        if config.enabled and config.strict:
            raise NativeKernelUnavailable(reason)
        decision = DispatchDecision("mlx_reference", reason, fallback_reason=reason)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("native_dispatch backend=mlx_reference reason=%s", reason)
        return decision

    if not config.enabled:
        return fallback("native_disabled")
    failure = request.capability_failure()
    if failure:
        return fallback(failure)
    # Qualification lookup is read-only, and explicit: no default directory scans.
    status = status_provider()
    if not status.get("loadable"):
        return fallback(status.get("reason", "extension_unavailable"))
    if request.mlx_build_sha256 != status.get("mlx_build_sha256"):
        return fallback("mlx_version_mismatch")
    candidate, failure = (registry or Registry()).match_exact(
        request,
        implementation_sha256=status["implementation_sha256"],
        binary_sha256=status["binary_sha256"],
    )
    if candidate is None:
        return fallback(failure)
    return DispatchDecision(
        "mlxboost_native",
        "qualified_exact_match",
        candidate_id=candidate.candidate_id,
        qualification_artifact=candidate.evidence_sha256,
        threadgroup_size=candidate.threadgroup_size,
    )
