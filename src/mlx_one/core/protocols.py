"""Structural native contracts without importing MLX or runtime services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from mlx_one.core.outputs import ModelOutput


@runtime_checkable
class NativeConfig(Protocol):
    @property
    def model_type(self) -> str: ...

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NativeConfig: ...


@runtime_checkable
class NativeState(Protocol):
    @property
    def offset(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    def reset(self) -> None: ...


@runtime_checkable
class NativeCausalModel(Protocol):
    """Minimum forward interface; cache layouts remain family-specific.

    Optimized logits selection, batching, paging and speculation are optional
    contracts, never inferred from structural conformance.
    """

    @property
    def config(self) -> NativeConfig: ...

    def make_cache(self) -> tuple[NativeState, ...]: ...

    def __call__(self, input_ids: Any, *, cache: tuple[Any, ...] | None = None) -> ModelOutput:
        ...
