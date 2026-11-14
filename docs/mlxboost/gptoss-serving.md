# Serve GPT-OSS-20B from the main checkout

The GPT-OSS MLXBoost implementation is a feature inside `mlx-one` 0.1.2, not
a separate accelerator package. Install from Git in your already activated
`mlx` environment after these changes have been pushed and merged to `main`:

```bash
unset PYTHONPATH
python -m pip install --upgrade --force-reinstall \
  "mlx-one[server] @ git+https://github.com/SSusantAchary/mlx-one.git@main"
python -m pip check
mlx-one --version
```

For later updates, run `mlx-one upgrade`. Preview with `mlx-one upgrade --dry-run`;
use `--ref` for a pushed branch or commit. Local unpushed changes are not included
in a Git installation. See [all upgrade options](../getting-started.md#upgrade-from-git).

The local package, Python API and CLI identify as 0.1.2. This aligns the older
checkout's labels to the maintainer's existing version; it is not a new release
or PyPI upload. The feature uses the existing pinned MLX/Metal 0.32.2 pairing.

Python configuration uses the normal package API:

```python
from mlx_one import CacheConfig, Engine, MLXBoostConfig

with Engine(
    "/path/to/gpt-oss-20b-MXFP4-Q4",
    offline=True,
    inference_mode="mlxboost_b1",
    cache_config=CacheConfig(
        prefix_cache_enabled=False, key_type="bf16", value_type="bf16"
    ),
    mlxboost_config=MLXBoostConfig(gptoss_mlx_lm=True),
) as engine:
    for chunk in engine.stream("Explain unified memory in one sentence."):
        print(chunk.text, end="", flush=True)
```

Stop other GPU model jobs before starting the server:

```bash
mlx-one serve "$HOME/mlx-one-models/gpt-oss-20b-MXFP4-Q4" \
  --offline --host 127.0.0.1 --port 8888 --alias gpt-oss-20b \
  --context-length 131072 --inference-mode mlxboost_b1 \
  --mlxboost-gptoss-mlx-lm --parallel 1 --prefill-chunk-size 2048 \
  --prefix-cache-mib 0 --no-cache-prompt \
  --cache-type-k bf16 --cache-type-v bf16 --timeout 0
```

The UI is at `http://127.0.0.1:8888`, with OpenAI-compatible API base
`http://127.0.0.1:8888/v1` and model alias `gpt-oss-20b`.

```bash
curl http://127.0.0.1:8888/health
curl http://127.0.0.1:8888/v1/models
```

The recipe is opt-in and experimental. It follows pinned MLX-LM expert dispatch
and compiled activation on the tested M4 / 32 GiB / mixed MXFP4-Q4 profile.
It changes arithmetic relative to the old eager path. Prefix reuse remains
disabled; choose this recipe separately from conservative prefill sorting and
`fast_decode`. The 128K context setting is an admission limit, not a latency or
all-request memory guarantee. Long-context native throughput evidence does not
by itself qualify the full HTTP chat/tool path at that size.

Integration checks in this checkout: 76 focused CLI/configuration/dispatch tests
passed; source imports and the installed CLI flag were verified with
`PYTHONPATH` unset; `pip check` found no broken requirements. No model was loaded
and no serving process was started during these integration checks, honoring
the requested manual launch. Previously recorded hardware measurements were
made in the isolated acceleration worktree and are not new measurements of
this checkout's other existing changes.
