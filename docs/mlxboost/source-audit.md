# MLXBoost source capture — 2026-11-01

The initial MLX-One HEAD is `c0bcfba3b0d19cd50a6b547ab170e89592083438`.
The owner had a modified pyproject and untracked benchmark, release and validation
files. Their status, binary diff and production hashes were saved before runtime
changes at `local/mlxboost/runs/audit-20261101/prechange/`. No owner files were reset.

| Source | Verified Git commit | License |
| --- | --- | --- |
| MLX baseline | `1f8e74e3f12f31365464a6867c6579f0e9b29d85` | MIT |
| MLX comparison | `36969c094354086e15d2a327a9c50feb20b9cf11` | MIT |
| MLX-LM reference | `93e6d4a6447b3080861985947f0820e76fcb957d` | MIT |
| oMLX study | `9de79bc39a01c1f3fec1e849903aaf0dd5478d7a` | Apache-2.0 |

`benchmarks/mlxboost/capture.py sources --fetch` fetched these exact objects into
four isolated repositories. `git cat-file -e` and `git rev-parse --verify` succeeded
for every object. The packaged `mlxboost/provenance/sources.yaml` lists extracted
file SHA-256 hashes; original bytes, repository trees and licenses are retained in
`local/mlxboost/sources/`. Every implementation path requested by the master prompt
exists. oMLX has no root NOTICE; this absence is recorded, not replaced with a
fabricated notice. The quant_matmul C++ subtree has a separate
`omlx/custom_kernels/quant_matmul/csrc/MLX_LICENSE.txt` (Apple MIT), captured and
hashed independently; the root Apache license is not treated as covering those
MLX-derived portions. No oMLX code is redistributed. The existing local reference
directory is not a Git checkout;
its Qwen2, Qwen3, Llama, LFM2, base and cache files are byte-identical to these
newly verified MLX-LM files. No Transformers source was used.

The MLX-LM generation reference is `generate.py::generate_step` (313 onward),
`_step` (403), prefill (437 onward), and async token pipeline (466–478).
Greedy sampling is `sample_utils.py::greedy_sampler` (133). Dense KV is
`models/cache.py::KVCache` (339). BPE initial-space removal is
`tokenizer_utils.py::BPEStreamingDetokenizer._maybe_trim_space` (230).

MLX `quantized.cpp::dispatch_qmv` (1739 in 0.32.2) selects stock vector kernels;
`nn.QuantizedLinear` already invokes the packed primitive. The 0.32.4 diff adds
row-contiguous early return, partial-output-row fast QMV handling and other
dispatch changes. None establishes a measured M4 gain. Qwen projection output
widths are aligned, so partial-row specialization is not evidence of their gap.
oMLX `fast.py::is_native_available` and optional binding checks demonstrate an
availability/fallback boundary; its M5/NAX specializations do not establish M4
eligibility. No custom kernel or upstream extension is adopted.

The installed baseline and development MLX/Metal are 0.32.2. Competitor MLX/Metal
are 0.32.4; oMLX has a distinct MLX-LM development dependency. Python is 3.12.11,
Apple M4/32 GiB/macOS 15.8.1. Metal is unavailable inside the sandbox and works in
the approved host execution context. The original environments were reused;
setup did not reinstall dependencies. Source study, package version identity,
numerical parity and model qualification are separate evidence claims.
