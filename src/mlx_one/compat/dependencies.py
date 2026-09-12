"""Actionable errors for optional compatibility dependencies."""

LEGACY_MLX_LM_INSTALL = "pip install 'mlx-one[legacy-mlx-lm]'"


def legacy_mlx_lm_error(feature: str) -> RuntimeError:
    """Return the stable error used by legacy mlx-lm bridges."""
    return RuntimeError(f"{feature} requires: {LEGACY_MLX_LM_INSTALL}")
