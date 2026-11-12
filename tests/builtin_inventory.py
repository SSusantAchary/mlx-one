"""Reviewed built-in inventory shared by registry regression checks."""

import json
from pathlib import Path

EXPECTED_MODELS = frozenset(
    json.loads(
        (
            Path(__file__).resolve().parents[1] / "scripts/release/expected_model_types.json"
        ).read_text()
    )
)
