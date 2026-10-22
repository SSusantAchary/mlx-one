"""P0.1 isolated native math experiments; discovery never initializes Metal."""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMMANDS = {
    "baseline": "qualification.p01_math",
    "run": "qualification.p01_experiments",
    "micro": "qualification.p01_micro",
    "compare": "qualification.p01_comparison",
    "external": "qualification.p01_external",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"--help", "-h"}:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=tuple(COMMANDS))
        parser.print_help()
        return 0
    command = argv.pop(0)
    if command not in COMMANDS:
        raise SystemExit("unsupported native math command; use --help")
    return import_module(COMMANDS[command]).main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
