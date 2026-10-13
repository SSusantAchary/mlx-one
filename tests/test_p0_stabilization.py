"""P0 regressions use the same instrumented native-model fixture as batching."""

import hashlib
import json
import sys

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one.p0_qualification import dense_snapshot_parity, inventory, p0_profiles, write_inventory


def test_p0_inventory_is_offline_and_never_qualifies(tmp_path):
    proof = inventory(tmp_path, checkpoints=[("missing", tmp_path / "absent")])
    assert proof["blockers"][0]["reason"] == "local checkpoint missing"
    assert proof["hardware_qualification"] == "pending"
    assert not proof["qualification_promoted"]
    write_inventory(tmp_path / "run", proof)
    checksums = json.loads((tmp_path / "run/checksums.json").read_text())
    assert checksums["readiness.json"] == hashlib.sha256(
        (tmp_path / "run/readiness.json").read_bytes()).hexdigest()
    assert all(p.slots == 1 and p.cache_type == "f16" for p in p0_profiles())


@pytest.mark.parametrize("length,boundary", [(2, 1), (33, 32), (257, 256)])
def test_dense_full_cached_and_trimmed_oracle(native, length, boundary):
    sys.modules["mlx.core"].float32 = np.float32
    proof = dense_snapshot_parity(native, [4] * length, boundary)
    assert proof["passed"] and proof["donor_position"] == length
    assert all(row["offsets"] == [length + i] * 2 for i, row in enumerate(proof["rows"]))
