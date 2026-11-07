"""Launch the existing CLI from explicit local source in an isolated environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if __name__ == "__main__":
    from mlx_one.cli import main

    main()
