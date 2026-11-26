"""Optional Phase 0 demo. Import is backend-free; models are never patched."""

from __future__ import annotations

from .capabilities import native_status
from .schemas import NativeConfig, NativeKernelUnavailable

__all__ = [
    "axpby",
    "experimental_axpby",
    "native_status",
    "NativeConfig",
    "NativeKernelUnavailable",
]


def reference_axpby(x, y, alpha=1.0, beta=1.0, *, stream=None):
    import mlx.core as mx

    # Explicit same-dtype scalars and separate products specify FP16 rounding.
    return mx.add(
        mx.multiply(x, mx.array(alpha, dtype=x.dtype), stream=stream),
        mx.multiply(y, mx.array(beta, dtype=y.dtype), stream=stream),
        stream=stream,
    )


def experimental_axpby(
    x, y, alpha=1.0, beta=1.0, *, enabled=False, threadgroup_size=128, stream=None
):
    """Explicit isolated developer call, always EXPERIMENTAL and inference-only."""
    if enabled is not True:
        raise NativeKernelUnavailable("experimental_opt_in_required")
    from .capabilities import load_extension

    return load_extension().axpby(
        x, y, alpha, beta, threadgroup_size=threadgroup_size, stream=stream
    )


def request_for(x, y, alpha=1.0, beta=1.0, *, stream=None):
    import platform

    import mlx.core as mx

    from .capabilities import distribution_version, load_extension, mlx_build_sha256
    from .schemas import KernelRequest

    if x.shape != y.shape:
        raise NativeKernelUnavailable("unsupported_shape")
    if x.dtype != y.dtype:
        raise NativeKernelUnavailable("unsupported_dtype")
    module = load_extension()
    layouts = [module.layout(t) for t in (x, y)]
    info = mx.device_info()
    # CPU stream selection is rejected in C++ too; no stream identity/pointers in evidence.
    device = stream.device if isinstance(stream, mx.Stream) else stream
    device = mx.default_device() if device is None else device
    return KernelRequest(
        chip=info.get("device_name", "unknown"),
        gpu_architecture=info.get("architecture", "unknown"),
        os_version=platform.mac_ver()[0],
        mlx_version=distribution_version("mlx"),
        mlx_metal_version=distribution_version("mlx-metal"),
        mlx_build_sha256=mlx_build_sha256(),
        dtype=str(x.dtype).split(".")[-1],
        shape=tuple(x.shape),
        strides=tuple(layouts[0]["strides"]),
        layout="row_contiguous" if all(d["row_contiguous"] for d in layouts) else "strided",
        stream_device="gpu" if device == mx.gpu else "cpu",
        alpha=float(alpha),
        beta=float(beta),
    )


def axpby(
    x, y, alpha=1.0, beta=1.0, *, config=None, registry=None, stream=None, return_decision=False
):
    """Safe demonstration dispatch. Default OFF; an unapproved tuner run cannot enable it."""
    from .dispatcher import choose
    from .schemas import DispatchDecision, KernelRequest

    config = NativeConfig.from_env() if config is None else config
    if not config.enabled:
        decision = choose(KernelRequest(), config)
    else:
        try:
            request = request_for(x, y, alpha, beta, stream=stream)
            decision = choose(request, config, registry)
        except NativeKernelUnavailable as exc:
            if config.strict:
                raise
            decision = DispatchDecision("mlx_reference", exc.reason, fallback_reason=exc.reason)
    if decision.backend == "mlxboost_native":
        result = experimental_axpby(
            x,
            y,
            alpha,
            beta,
            enabled=True,
            threadgroup_size=decision.threadgroup_size,
            stream=stream,
        )
    else:
        result = reference_axpby(x, y, alpha, beta, stream=stream)
    return (result, decision) if return_decision else result
