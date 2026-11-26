"""Actual GPU tests; never count unavailable hardware as a pass."""

import os

import pytest


@pytest.fixture(scope="module")
def gpu():
    if os.environ.get("MLXBOOST_PHASE0_GPU_TESTS") != "1":
        pytest.skip("set MLXBOOST_PHASE0_GPU_TESTS=1 in a Metal-enabled session")
    from mlx_one.mlxboost.native import native_status

    status = native_status()
    if not status["loadable"]:
        pytest.skip("native extension unavailable: " + status["reason"])
    import mlx.core as mx

    from mlx_one.mlxboost.native.capabilities import load_extension

    return mx, load_extension()


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(0,), (), (1,), (63,), (64,), (65,), (257,), (2, 3), (4096,)])
@pytest.mark.parametrize("threads", [64, 128, 256])
def test_native_parity_and_stream_lifetime(gpu, dtype, shape, threads):
    mx, module = gpu
    from mlx_one.mlxboost.native import reference_axpby

    mx.random.seed(7)
    stream = mx.new_stream(mx.gpu)
    x = mx.random.uniform(-1, 1, shape).astype(getattr(mx, dtype))
    y = mx.random.uniform(-1, 1, shape).astype(getattr(mx, dtype))
    expected = reference_axpby(x, y, 0.75, -0.25, stream=stream)
    actual = module.axpby(x, y, 0.75, -0.25, threadgroup_size=threads, stream=stream)
    del x, y
    mx.eval(actual, expected)
    tol = 1e-3 if dtype == "float16" else 1e-6
    assert bool(mx.allclose(actual, expected, atol=tol, rtol=tol).item())
    assert actual.shape == shape
    assert actual.dtype == getattr(mx, dtype)


def test_edge_cases_and_transform_rejection(gpu):
    mx, module = gpu
    from mlx_one.mlxboost.native import reference_axpby

    x = mx.array([float("nan"), float("inf"), -float("inf"), 0])
    y = mx.ones((4,))
    actual, expected = module.axpby(x, y, 1, 1), reference_axpby(x, y)
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, equal_nan=True).item())
    for bad in (mx.ones((4,), dtype=mx.int32), mx.ones((4,), dtype=mx.bfloat16)):
        with pytest.raises(ValueError, match="unsupported_dtype"):
            module.axpby(bad, bad, 1, 1)
    with pytest.raises(ValueError, match="unsupported_shape"):
        module.axpby(mx.ones((3,)), mx.ones((4,)), 1, 1)
    strided = mx.arange(16, dtype=mx.float32).reshape(4, 4).T
    mx.eval(strided)
    with pytest.raises(ValueError, match="unsupported_layout"):
        module.axpby(strided, strided, 1, 1)
    with pytest.raises(ValueError, match="unsupported_stream"):
        module.axpby(y, y, 1, 1, stream=mx.cpu)
    with pytest.raises(ValueError, match="threadgroup"):
        module.axpby(y, y, 1, 1, threadgroup_size=2048)

    def fn(a):
        return module.axpby(a, a, 1, 1)

    for transform in (
        lambda: mx.grad(lambda a: mx.sum(fn(a)))(y),
        lambda: mx.jvp(fn, [y], [y]),
        lambda: mx.vjp(fn, [y], [y]),
        lambda: mx.vmap(fn)(mx.ones((2, 4))),
    ):
        with pytest.raises((ValueError, RuntimeError), match="inference-only"):
            transform()


def test_default_and_unqualified_fallback_and_repeated_execution(gpu):
    mx, module = gpu
    from mlx_one.mlxboost.native import NativeConfig, NativeKernelUnavailable, axpby

    x = mx.arange(257, dtype=mx.float32)
    for config, reason in (
        (NativeConfig(), "native_disabled"),
        (NativeConfig(True), "missing_evidence"),
    ):
        actual, decision = axpby(x, x, config=config, return_decision=True)
        mx.eval(actual)
        assert decision.fallback_reason == reason
        assert bool(mx.all(actual == 2 * x).item())
    with pytest.raises(NativeKernelUnavailable, match="missing_evidence"):
        axpby(x, x, config=NativeConfig(True, True))
    for _ in range(100):
        mx.eval(module.axpby(x, x, 1, 1))


def test_offline_tuning_records_three_candidates_without_approval(gpu, tmp_path):
    from mlx_one.mlxboost.native import request_for
    from mlx_one.mlxboost.native.autotune import export_proposal, tune
    from mlx_one.mlxboost.native.registry import Registry

    mx, _ = gpu
    path, record = tune(tmp_path, length=4096, trials=10, warmup=3)
    assert {r["threadgroup_size"] for r in record["results"]} == {64, 128, 256}
    assert record["approved_for_runtime"] is False
    assert all(r["correct"] for r in record["results"])
    proposal = export_proposal(path, record)
    x = mx.ones((4096,))
    request = request_for(x, x, 0.75, -0.25)
    _, reason = Registry(proposal).match_exact(
        request,
        implementation_sha256=record["implementation_sha256"],
        binary_sha256=record["binary_sha256"],
    )
    assert reason in {"regression_blocked", "unapproved_evidence", "missing_evidence"}


def test_concurrent_native_streams_and_request_local_fallback(gpu):
    from concurrent.futures import ThreadPoolExecutor

    from mlx_one.mlxboost.native import NativeConfig, axpby

    mx, module = gpu
    x = mx.arange(257, dtype=mx.float32)
    mx.eval(x)

    def run(index):
        stream = mx.new_stream(mx.gpu)
        if index % 2:
            out = module.axpby(x, x, 1, 1, stream=stream)
            backend = "explicit_experimental"
        else:
            out, decision = axpby(x, x, config=NativeConfig(), stream=stream, return_decision=True)
            backend = decision.backend
        mx.eval(out)
        mx.synchronize(stream)
        assert bool(mx.all(out == 2 * x).item())
        return backend

    with ThreadPoolExecutor(4) as pool:
        backends = list(pool.map(run, range(24)))
    assert backends == ["explicit_experimental" if i % 2 else "mlx_reference" for i in range(24)]
