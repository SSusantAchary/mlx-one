"""Architecture declarations, separate from checkpoint/precision qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_one.core.registry import ModelRegistration


@dataclass(frozen=True)
class ArchitectureCapabilities:
    model_type: str
    modality: str
    operations: tuple[str, ...]
    tasks: tuple[str, ...]
    loader_declared: bool
    weight_contract_declared: bool
    qualification: str = "NOT_QUALIFIED"

    def supports(self, operation: str) -> bool:
        return operation in self.operations or operation in self.tasks

    def to_dict(self) -> dict:
        return asdict(self)


def describe_architecture(registration: ModelRegistration) -> ArchitectureCapabilities:
    """Inspect lazy metadata; do not resolve model classes or claim precision support."""
    return ArchitectureCapabilities(
        registration.model_type,
        registration.modality,
        tuple(sorted(registration.capabilities)),
        tuple(sorted(registration.tasks)),
        registration.loader_path is not None,
        registration.weight_contract_path is not None,
    )
