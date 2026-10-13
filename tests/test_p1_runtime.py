"""Backend-free P1 contracts and differential runtime regressions."""

from mlx_one.p1_profiles import CHECKPOINTS, REFERENCE_PINS


def test_p1_pins_are_immutable_and_converted_models_are_explicit():
    assert len(CHECKPOINTS) == 6 and len(REFERENCE_PINS) == 3
    assert all(len(c.revision) == 40 for c in CHECKPOINTS.values())
    assert [name for name, c in CHECKPOINTS.items() if c.conversion_required] == [
        "smol-small", "smol-large"]
    assert CHECKPOINTS["smol-small"].context_tokens == 8192
