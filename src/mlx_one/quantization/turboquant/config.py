"""Backend-free, lossless configuration for the supported on-disk TQ3 format."""

from collections.abc import Mapping
from dataclasses import dataclass

from mlx_one.core.quantization import UnsupportedQuantization

CODEBOOK = (
    -2.15234375,
    -1.34375,
    -0.755859375,
    -0.2451171875,
    0.2451171875,
    0.755859375,
    1.34375,
    2.15234375,
)


@dataclass(frozen=True)
class TQ3Storage:
    word_dtype: str = "uint32"
    indices_per_word: int = 10
    bit_order: str = "lsb_first"
    scale_dtype: str = "float16"
    orientation: str = "expert_output_input"


@dataclass(frozen=True)
class TQ3Config:
    mode: str = "turboquant"
    bits: int = 3
    group_size: int = 64
    codebook: tuple[float, ...] = CODEBOOK
    rotation: str = "hadamard"
    rotation_seed: int = 42
    fuse_rotations: bool = False
    use_qjl: bool = False
    overrides: tuple = ()
    storage: TQ3Storage = TQ3Storage()

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise UnsupportedQuantization("TQ3 metadata must be an object")
        fields = {
            "mode",
            "bits",
            "group_size",
            "codebook",
            "rotation",
            "rotation_seed",
            "fuse_rotations",
            "use_qjl",
        }
        if fields - set(data):
            raise UnsupportedQuantization(f"missing TQ3 metadata: {sorted(fields - set(data))}")
        if (
            data["mode"] != "turboquant"
            or type(data["bits"]) is not int
            or data["bits"] != 3
            or type(data["group_size"]) is not int
            or data["group_size"] != 64
            or data["rotation"] != "hadamard"
            or type(data["rotation_seed"]) is not int
            or data["fuse_rotations"] is not False
            or data["use_qjl"] is not False
        ):
            raise UnsupportedQuantization(
                "unsupported TQ3 variant: requires 3-bit/g64 Hadamard, no QJL or fused rotations"
            )
        try:
            codebook = tuple(data["codebook"])
        except TypeError as exc:
            raise UnsupportedQuantization("invalid TQ3 codebook") from exc
        if codebook != CODEBOOK or any(type(v) not in (int, float) for v in codebook):
            raise UnsupportedQuantization("unsupported or corrupt TQ3 codebook")
        overrides = []
        for path, spec in data.items():
            if path in fields:
                continue
            if not isinstance(path, str) or not path or any(not x for x in path.split(".")):
                raise UnsupportedQuantization(f"unknown TQ3 setting: {path}")
            if spec is False:
                overrides.append((path, False))
            elif isinstance(spec, Mapping):
                from mlx_one.core.quantization import quantization_spec

                overrides.append((path, quantization_spec(spec)))
            else:
                raise UnsupportedQuantization(f"invalid TQ3 override: {path}")
        return cls(rotation_seed=data["rotation_seed"], overrides=tuple(overrides))

    def to_dict(self):
        return dict(
            mode=self.mode,
            bits=self.bits,
            group_size=self.group_size,
            codebook=list(self.codebook),
            rotation=self.rotation,
            rotation_seed=self.rotation_seed,
            fuse_rotations=False,
            use_qjl=False,
            **dict(self.overrides),
        )
