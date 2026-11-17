# G01/G02 source mapping appendix

The maps record the bytes actually inspected in this dirty worktree, SHA256, source spans, provenance and limitations: [G01 map](mlxboost-g01-source-map.json), [G02 map](mlxboost-g02-source-map.json). Each map contains 42 inspected file hashes; backend-free validation checked all hashes and line bounds. That check validates documentation provenance, not GPU compatibility or mathematical parity.

| Reference | Pinned identity and purpose |
| --- | --- |
| Local MLX-One | HEAD `4cad1634562c5c12933882deb0b8ab3c9461dd97` plus individual worktree file hashes; Qwen2/2.5, Qwen3, Llama, LFM2, Gemma3-text, GPT-OSS and shared activation/math sources. Actual local equations govern eligibility |
| MLX | [v0.32.2](https://github.com/ml-explore/mlx/tree/v0.32.2), captured revision `1f8e74e3f12f31365464a6867c6579f0e9b29d85`; `rms_norm.metal`, `fast.cpp`, activation/sigmoid, extension stream/build contracts. Installed 0.32.2 headers/Python sources are separately hashed rather than misrepresented as fetched tag files |
| MLX-LM | Captured revision `93e6d4a6447b3080861985947f0820e76fcb957d`; shared compiled activation and concrete family graphs. Qwen3's alternative local port records its distinct `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd` / v0.31.3 lineage |
| ZMLX | [d8cd0d88d4299ca6821d706f9d5b4a3520d688f5](https://github.com/Hmbown/ZMLX/tree/d8cd0d88d4299ca6821d706f9d5b4a3520d688f5); actual `transformer.py`, `norms.py`, `msl.py`, `metal.py`, residual/SwiGLU patch patterns, selector fidelity exclusions, kernel/benchmark documents and MIT license |
| Unpinned local reference archive | `references_libs/mlx-lm-main/.../norm_add_norm.py`; hash recorded, upstream revision unknown. Different residual contract, comparison only; never imported as runtime |

MLX, MLX-LM and the inspected ZMLX are MIT-licensed. Future substantial adaptation must preserve the applicable notices and exact provenance. This design assignment copies no production shader and introduces no foreign runtime dependency. Upstream benchmark claims are not results measured on this host. Missing source captures and provenance limitations remain explicit in each detailed study/map.
