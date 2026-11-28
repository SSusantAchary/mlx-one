# Author: Susant Achary
"""Load the checked-in Metal fragments used by MLX's JIT compiler."""

from functools import cache
from importlib.resources import files
from string import Template

_FRAGMENT_MARKER = "// Begin shader source\n"


@cache
def _read_fragment(name: str) -> str:
    text = files("mlx_one").joinpath("metal_sources", name).read_text(encoding="utf-8")
    _, marker, source = text.partition(_FRAGMENT_MARKER)
    if not marker:
        raise ValueError(f"Metal fragment has no source marker: {name}")
    return source


def metal_source(name: str, **fragments: str) -> str:
    """Read a Metal fragment and fill its named runtime specializations."""
    source = _read_fragment(name)
    if not fragments:
        return source
    return Template(source).substitute(fragments)
