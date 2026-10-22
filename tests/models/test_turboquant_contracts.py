"""Header/packing contracts are always tested, including without Metal."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mlx_one.core.quantization import UnsupportedQuantization, quantization_config
from mlx_one.models.language.gpt_oss.config import GptOssConfig
from mlx_one.quantization.turboquant.config import CODEBOOK, TQ3Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/turboquant"))
from oracle import pack, reconstruct, rotate_power2  # noqa: E402

METADATA = ROOT / "tests/fixtures/turboquant_metadata"


def test_real_metadata_and_headers():
    data = json.loads((METADATA / "config.json").read_text())
    config = GptOssConfig.from_dict(data)
    assert (config.num_hidden_layers, config.num_local_experts, config.num_experts_per_tok) == (
        24,
        32,
        4,
    )
    assert config.head_dim == 64 and config.rope_scaling["truncate"] is False
    q = quantization_config(data["quantization"])
    assert q == quantization_config(data["quantization_config"]) == TQ3Config.from_dict(q).to_dict()
    assert TQ3Config().storage.indices_per_word == 10
    headers = {
        k: v
        for p in METADATA.glob("*.header.json")
        for k, v in json.loads(p.read_text()).items()
        if k != "__metadata__"
    }
    index = json.loads((METADATA / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(headers) == set(index) and len(headers) == 966
    for key, value in headers.items():
        if key.endswith(".codebook"):
            prefix = key[:-9]
            assert value["shape"] == [8] and value["dtype"] == "F16"
            weight, scales, signs = [
                headers[prefix + "." + n] for n in ("weight", "scales", "signs")
            ]
            k = signs["shape"][0]
            assert weight["dtype"] == "U32"
            assert weight["shape"][-1] == (k + 9) // 10
            assert scales["shape"] == weight["shape"][:-1] + [k // 64]


@pytest.mark.parametrize(
    "change",
    [
        dict(bits=4),
        dict(group_size=32),
        dict(use_qjl=True),
        dict(rotation="none"),
        dict(fuse_rotations=True),
        dict(rotation_seed=True),
        dict(codebook=[0] * 8),
        dict(bits=True),
        dict(unknown="wrong"),
    ],
)
def test_bad_quantization_metadata(change):
    with pytest.raises(UnsupportedQuantization):
        quantization_config({**TQ3Config().to_dict(), **change})


def test_all_codes_padding_group_boundaries_and_experts():
    idx = np.arange(2 * 3 * 128, dtype=np.uint32).reshape(2, 3, 128) % 8
    words = pack(idx)
    assert words.shape == (2, 3, 13)
    assert words[0, 0, 0] & 7 == 0 and (words[0, 0, 0] >> 27) & 7 == 1
    scales = np.arange(1, 13, dtype=np.float16).reshape(2, 3, 2)
    cb = np.array(CODEBOOK)
    expected = cb[idx] * np.repeat(scales, 64, axis=-1)
    np.testing.assert_array_equal(reconstruct(words, scales, cb, 128), expected)
    # Two high bits and final padding slots are not real weights.
    words |= np.uint32(3 << 30)
    words[..., -1] |= np.uint32(63 << 24)
    np.testing.assert_array_equal(reconstruct(words, scales, cb, 128), expected)
    for col in (0, 9, 10, 63, 64, 127):
        assert expected[1, 2, col] == cb[idx[1, 2, col]] * scales[1, 2, col // 64]
    with pytest.raises(ValueError):
        reconstruct(words.astype(np.int32), scales, cb, 128)


def test_known_rotation_inverse():
    x = np.arange(64, dtype=float).reshape(1, 64) / 64
    signs = np.where(np.arange(64) % 3, 1, -1)
    rot = rotate_power2(x, signs)
    # Inverse is H then signs, not the forward transform applied twice.
    inverse = rotate_power2(rot, np.ones(64)) * signs
    np.testing.assert_allclose(inverse, x, atol=1e-14)


def test_format_label_and_dispatch_boundaries():
    from mlx_one.core.precision import describe_precision
    from mlx_one.quantization.turboquant.dispatch import dense_kernel, expert_kernel
    from mlx_one.schemas import ModelSpec

    model = ModelSpec(
        model_id="manjunathshiva/gpt-oss-20b-tq3",
        revision="a" * 40,
        quantization=TQ3Config().to_dict(),
    )
    view = describe_precision(model)
    assert view["weight"]["kind"] == "turboquant_tq3"
    assert view["weight"]["label"] == "TQ3 / TurboQuant 3-bit"
    assert view["kv_cache"]["dtype"] is None and view["qualification"] == "NOT_QUALIFIED"
    assert dense_kernel(1, 4096) == "qmv"
    assert dense_kernel(2, 2880) == "qmv_rows"
    assert dense_kernel(2, 4096) == "qmm"
    assert dense_kernel(8, 512) == "qmm"
    assert expert_kernel(127, 2880) == "gather_qmv"
    assert expert_kernel(128, 2880) == "gather_qmm"
    assert expert_kernel(511, 67) == "gather_qmv"
    assert expert_kernel(512, 67) == "gather_qmm"
