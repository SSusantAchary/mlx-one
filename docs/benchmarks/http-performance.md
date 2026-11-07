# Historical GPT-OSS HTTP serving

Measured on Apple M4 with 32 GB unified memory, Python 3.12, using the native MLX-One server on 2026-10-23. One warmup followed by five sequential tool-call requests; 175 prompt tokens and 91 generated tokens per request. Throughput and TTFT are per-column medians; memory is the observed peak. This serving workload uses a separate protocol from the direct-model text performance tables.

| Model name | Parameters | Configured context (tokens) | Quantization/precision | Prefill tokens/sec | Decode tokens/sec | Native first-token time (ms) | Peak MLX memory (GiB) |
|---|---|---:|---|---:|---:|---:|---:|
| [mlx-community/gpt-oss-20b-MXFP4-Q4](../validation/gpt-oss-kilo-2026-10-23.json) | 20B / 3.6B active | 131,072 | Mixed MXFP4 / affine Q4–Q8 | **75.95** | **18.96** | 2355.38 | 10.608 |

Native first-token time includes hidden analysis; the median first actionable tool delta arrived at **7.127 seconds**, after argument validation. The configured 131,072-token context was not tested at its full length. See the [full measurements and methodology](../GPT_OSS_KILO_COMPATIBILITY.md#measured-performance). Standardized direct-model 128/128 coverage remains unchanged.

## Relationship to current checkpoint status

The [current inventory](inventory.md) pins separate MXFP4/Q4 and
TQ3 configurations whose full-checkpoint direct-model runs remain pending. This
2026-10-23 report used a local model path; its immutable checkpoint revision and
MLX version are **unavailable**. The [chart provenance](figures/README.md)
records this limitation. A historical working serving case cannot identify or
qualify either currently pinned checkpoint. No inventory status is changed.

The raw report records hardware, request contents, actual token counts, cache
settings, concurrency and source/build identifiers where available. Consult the
[compatibility report](../GPT_OSS_KILO_COMPATIBILITY.md) and
[raw request records](../validation/gpt-oss-kilo-2026-10-23.json). Unrecorded
checkpoint identity, software versions and precision provenance remain unavailable;
no replacement metadata is inferred. This HTTP table is excluded from direct-model
comparison cohorts.
