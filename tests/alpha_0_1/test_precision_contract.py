"""Metadata proof must never imply real weight, compute or hardware proof."""

import json
import struct

import pytest

from mlx_one.core.precision import describe_precision, validate_finite_safetensors
from mlx_one.schemas import ModelSpec, Precision


def spec(**kwargs):
    return ModelSpec(model_id="org/model", revision="a" * 40, **kwargs)


@pytest.mark.parametrize("dtype,kind", [("float32", "fp32"), ("float16", "fp16"),
                                        ("bfloat16", "bf16")])
def test_float_storage_does_not_infer_compute_or_cache(dtype, kind):
    view = describe_precision(spec(dtype=dtype))
    assert view["weight"]["kind"] == kind
    assert view["activation"]["dtype"] is None
    assert view["kv_cache"]["dtype"] is None
    assert view["qualification"] == "NOT_QUALIFIED"


@pytest.mark.parametrize("bits", [4, 8])
def test_affine_metadata_and_explicit_compute_are_separate(bits):
    model = spec(dtype="float16", quantization={"bits": bits, "group_size": 64})
    before = model.to_dict()
    view = describe_precision(model, activation_dtype="bf16", kv_cache_dtype="fp16")
    assert view["weight"]["kind"] == f"mlx_affine_q{bits}"
    assert view["activation"]["dtype"] == "bf16"
    assert view["kv_cache"]["dtype"] == "fp16"
    assert model.to_dict() == before  # no serialized schema change or mutation
    assert ModelSpec.from_dict(before) == model


@pytest.mark.parametrize("quant", [{"bits": True}, {"bits": 8.0}, {"bits": 3},
    {"bits": 4, "group_size": False}, {"bits": 8, "group_size": 0},
    {"bits": 4, "mode": "mxfp4"}, {"bits": 8, "quant_method": "gguf"}])
def test_fake_affine_metadata_rejected(quant):
    with pytest.raises(ValueError):
        describe_precision(spec(dtype="float16", quantization=quant))


def test_integer_planning_label_is_not_an_affine_checkpoint():
    assert Precision.INT8.value == "int8"  # public planning enum preserved
    with pytest.raises(ValueError, match="floating dtype"):
        describe_precision(spec(dtype="int8"))


def test_mixed_exceptions_are_retained_and_unobserved_dtypes_rejected():
    quant = {"bits": 4, "group_size": 64, "layers.0": {"bits": 8}}
    view = describe_precision(spec(quantization=quant))
    assert view["weight"]["kind"] == "mixed"
    assert view["weight"]["metadata"] == quant
    with pytest.raises(ValueError, match="kv_cache"):
        describe_precision(spec(dtype="fp16"), kv_cache_dtype="q8_0")


@pytest.mark.parametrize("dtype,payload,valid", [
    ("F32", struct.pack("<f", 1.0), True),
    ("F16", struct.pack("<e", -2.0), True),
    ("BF16", struct.pack("<H", 0x3F80), True),
    ("U32", struct.pack("<I", 0xFFFFFFFF), True),
    ("F32", struct.pack("<f", float("nan")), False),
    ("F16", struct.pack("<e", float("inf")), False),
    ("BF16", struct.pack("<H", 0x7F80), False),
    ("BF16", struct.pack("<H", 0xFFC1), False),
])
def test_cpu_payload_validation_preserves_packed_bits_and_checks_bf16(
    tmp_path, dtype, payload, valid,
):
    header = json.dumps({"weight": {"dtype": dtype, "shape": [1],
                                  "data_offsets": [0, len(payload)]}}).encode()
    header += b" " * (-len(header) % 8)
    path = tmp_path / "model.safetensors"
    original = struct.pack("<Q", len(header)) + header + payload
    path.write_bytes(original)
    if valid:
        validate_finite_safetensors(path)
    else:
        with pytest.raises(ValueError, match="non-finite"):
            validate_finite_safetensors(path)
    assert path.read_bytes() == original


def test_malformed_checkpoint_uses_loader_compatible_value_error(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"malformed header")
    with pytest.raises(ValueError, match="invalid safetensors checkpoint"):
        validate_finite_safetensors(path)
