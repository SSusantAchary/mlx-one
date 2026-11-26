"""Real MLX fallback tests also run when the optional native extension is absent."""
import os

import pytest


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("length", [0, 1, 65, 4096])
def test_real_reference_fallback_without_qualification(dtype, length):
    if os.environ.get("MLXBOOST_PHASE0_GPU_TESTS") != "1":
        pytest.skip("explicit Metal session required")
    mx = pytest.importorskip("mlx.core")
    from mlx_one.mlxboost.native import NativeConfig, NativeKernelUnavailable, axpby

    x = mx.arange(length, dtype=getattr(mx, dtype))
    y = -x
    expected = .75 * x - .25 * y
    for config in (NativeConfig(), NativeConfig(True)):
        out, decision = axpby(x, y, .75, -.25, config=config, return_decision=True)
        mx.eval(out, expected)
        assert bool(mx.allclose(out, expected, atol=1e-3, rtol=1e-3).item())
        assert decision.backend == "mlx_reference"
        assert decision.reason in {"native_disabled", "extension_unavailable", "missing_evidence"}
    with pytest.raises(NativeKernelUnavailable):
        axpby(x, y, config=NativeConfig(True, True))
