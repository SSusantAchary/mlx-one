"""Compare matched, checksummed native and external artifacts without promotion."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.p01_comparison import main

if __name__ == "__main__":
    raise SystemExit(main())
