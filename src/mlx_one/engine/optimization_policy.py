"""Immutable request-scoped native math choices; no native imports or global patches."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass

POLICY_VERSION = "native-math-v6"


@dataclass(frozen=True)
class NativeMathPolicy:
    attention: str = "reference"
    projection: str = "reference"
    logits: str = "all"
    prefill_chunk_size: int | None = None
    hybrid: str = "reference"
    enable_compile: bool = False

    def __post_init__(self):
        for field, choices in (("attention", {"reference", "mlx_sdpa", "tiled_256",
                                                  "native_gather_tiled"}),
                               ("projection", {"reference", "mlx_qmm", "native"}),
                               ("logits", {"all", "last_only"}),
                               ("hybrid", {"reference", "fast"})):
            if getattr(self, field) not in choices:
                raise ValueError(f"unsupported native math {field}: {getattr(self, field)}")
        width = self.prefill_chunk_size
        if width is not None and (type(width) is not int or width not in {256, 512, 1024, 2048}):
            raise ValueError("prefill chunk size must be 256, 512, 1024, 2048 or None")
        if type(self.enable_compile) is not bool:
            raise ValueError("enable_compile must be boolean")

    @classmethod
    def optimized(cls):
        # New routes remain opt-in experiments. No qualification is inferred.
        return cls(logits="last_only")

    def to_dict(self):
        from mlx_one.engine.kernel_dispatch import MATH_KERNEL_VERSION

        return {"version": POLICY_VERSION, "kernel_version": MATH_KERNEL_VERSION,
                **asdict(self), "qualified": False,
                "attention_accumulation_operands": "float32"
                if self.attention in {"mlx_sdpa", "native_gather_tiled"} else "reference",
                "projection_accumulation_operands": "stock_fp16_vector_only"
                if self.projection == "mlx_qmm" else "reference"}


REFERENCE_MATH = NativeMathPolicy()
_math_context = ContextVar("mlx_one_native_math", default=None)


@dataclass(frozen=True)
class MathContext:
    policy: NativeMathPolicy = REFERENCE_MATH
    family: str | None = None
    phase: str | None = None
    observer: object = None


_REFERENCE_CONTEXT = MathContext()


def current_math():
    return _math_context.get() or _REFERENCE_CONTEXT


@contextmanager
def math_execution(policy, *, family=None, phase=None, observer=None):
    if not isinstance(policy, NativeMathPolicy):
        raise TypeError("NativeMathPolicy required")
    token = _math_context.set(MathContext(policy, family, phase, observer))
    try:
        yield
    finally:
        _math_context.reset(token)
