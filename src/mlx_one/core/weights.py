"""Explicit tensor-name and shape contracts for native architectures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class WeightContractError(ValueError):
    """Raised when supplied tensors do not match an architecture contract."""


@dataclass(frozen=True)
class WeightContract:
    expected: Mapping[str, tuple[int, ...]]
    optional: frozenset[str] = field(default_factory=frozenset)
    ignored_suffixes: tuple[str, ...] = ()

    def validate(self, weights: Mapping[str, Any]) -> None:
        filtered = {
            name: value
            for name, value in weights.items()
            if not any(name.endswith(suffix) for suffix in self.ignored_suffixes)
        }
        required = set(self.expected) - set(self.optional)
        missing = sorted(required - set(filtered))
        unexpected = sorted(set(filtered) - set(self.expected))
        mismatched = sorted(
            name
            for name in set(filtered) & set(self.expected)
            if tuple(filtered[name].shape) != tuple(self.expected[name])
        )
        problems = []
        if missing:
            problems.append(f"missing tensors: {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected tensors: {', '.join(unexpected)}")
        if mismatched:
            details = ", ".join(
                f"{name}={tuple(filtered[name].shape)} expected {self.expected[name]}"
                for name in mismatched
            )
            problems.append(f"shape mismatches: {details}")
        if problems:
            raise WeightContractError("; ".join(problems))
