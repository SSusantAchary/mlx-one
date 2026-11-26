"""Backend-free, immutable demo-kernel contracts. No model qualification authority."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

SCHEMA_VERSION = 1
KERNEL_ID = "mlxboost.axpby.v1"
MLX_VERSION = "0.32.2"
THREADGROUPS = (64, 128, 256)


class NativeKernelUnavailable(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class NativeConfig:
    enabled: bool = False
    strict: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.strict) is not bool:
            raise TypeError("native switches must be bool")

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        values = []
        for key in ("MLX_ONE_NATIVE", "MLX_ONE_NATIVE_STRICT"):
            value = env.get(key, "0")
            if value not in {"0", "1"}:
                raise ValueError(f"{key} must be 0 or 1")
            values.append(value == "1")
        return cls(*values)


@dataclass(frozen=True)
class KernelRequest:
    operation: str = KERNEL_ID
    model_architecture: str | None = None
    chip: str = "unknown"
    gpu_architecture: str = "unknown"
    os_version: str = "unknown"
    mlx_version: str = MLX_VERSION
    mlx_build_sha256: str = ""
    mlx_metal_version: str = MLX_VERSION
    dtype: str = "float32"
    shape: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    layout: str = "row_contiguous"
    quantization: str | None = None
    group_size: int | None = None
    batch: int = 1
    cache_mode: str = "none"
    execution_mode: str = "inference"
    stream_device: str = "gpu"
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or not isinstance(self.strides, tuple):
            raise TypeError("shape and strides must be immutable tuples")

    def capability_failure(self):
        if self.operation != KERNEL_ID or self.model_architecture is not None:
            return "reference_selected"
        if self.mlx_version != MLX_VERSION or self.mlx_metal_version != MLX_VERSION:
            return "mlx_version_mismatch"
        if self.dtype not in {"float16", "float32"}:
            return "unsupported_dtype"
        if (
            any(type(d) is not int or d < 0 for d in self.shape)
            or math.prod(self.shape) > 2**32 - 1
            or self.batch != 1
        ):
            return "unsupported_shape"
        expected, stride = [], 1
        for dim in reversed(self.shape):
            expected.append(stride)
            stride *= max(1, dim)
        if self.layout != "row_contiguous" or self.strides != tuple(reversed(expected)):
            return "unsupported_layout"
        if self.quantization is not None or self.group_size is not None:
            return "unsupported_quantization"
        if self.execution_mode != "inference" or self.cache_mode != "none":
            return "reference_selected"
        if self.stream_device != "gpu":
            return "unsupported_stream"
        if self.chip == "unknown" or self.gpu_architecture == "unknown":
            return "unsupported_chip"
        if not all(math.isfinite(v) for v in (self.alpha, self.beta)):
            return "unsupported_parameters"
        return None


@dataclass(frozen=True)
class KernelCandidate:
    candidate_id: str
    kernel_id: str
    semantic_version: str
    implementation_sha256: str
    binary_sha256: str
    threadgroup_size: int
    evidence_sha256: str


@dataclass(frozen=True)
class AutotuneCandidate:
    request: KernelRequest
    implementation_sha256: str
    binary_sha256: str
    threadgroup_size: int
    max_threads: int
    grid: str = "one_thread_per_element"
    tile: int = 1
    workspace_bytes: int = 0


@dataclass(frozen=True)
class DispatchDecision:
    backend: str
    reason: str
    fallback_reason: str | None = None
    candidate_id: str | None = None
    qualification_artifact: str | None = None
    threadgroup_size: int | None = None
    required_flags: tuple[str, ...] = ("MLX_ONE_NATIVE=1",)
    scope: str = "demo_operator_only"
