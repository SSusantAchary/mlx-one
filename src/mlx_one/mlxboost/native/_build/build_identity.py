"""Executed at configure time; never imports mlx.core or initializes Metal."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

root = Path(__file__).parent
digest = hashlib.sha256()
for path in sorted(root.iterdir()):
    if path.is_file() and path.suffix in {".py", ".cpp", ".mm", ".h", ".metal", ".txt"}:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
mlx_root = Path(next(iter(importlib.util.find_spec("mlx").submodule_search_locations)))
mlx_hash = hashlib.sha256((mlx_root / "lib/libmlx.dylib").read_bytes()).hexdigest()
if len(sys.argv) > 1:
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(schema_version=1, mlx_version="0.32.2",
                                     mlx_build_sha256=mlx_hash,
                                     implementation_sha256=digest.hexdigest()), sort_keys=True))
print(f'#define MLXBOOST_SOURCE_SHA "{digest.hexdigest()}"')
print(f'#define MLXBOOST_MLX_SHA "{mlx_hash}"')
print('#define MLXBOOST_MLX_VERSION "0.32.2"')
print(f'#define MLXBOOST_BUILD_PYTHON {json.dumps(__import__("sys").version.split()[0])}')
