# Cache Runtime M4/32 GB Qualification — Harness & Preflight Results

Date: 2026-09-16  
Host inspected: Apple M4, 32 GB unified memory, macOS 15.7.9.

## Execution status

The initial sandbox could not initialize Metal. Native tests were subsequently run outside that
sandbox on the inspected M4 host. This document remains a partial result: the complete parallel and
cancellation matrix is not yet run, and Cache Runtime M4 APC is not qualified.

The non-GPU cache, server, scheduler, allocator, and lifecycle suite passed: `357 passed, 14
skipped`. The qualification harness is [cache_runtime_bench.py](../benchmarks/cache_runtime_bench.py).

## Discovered locally cached candidates

| Model | Params | Architecture | Context | Cache topology | Runtime status |
| --- | ---: | --- | ---: | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 0.5B | Qwen2/GQA | config available; snapshot incomplete | ordinary attention KV per layer | NOT TESTED — no resolvable local snapshot |
| `mlx-community/LFM2-350M-4bit` | 350M | LFM2 hybrid | 128K | 16 layers: 6 attention KV (2, 5, 8, 10, 12, 14), 10 convolution states | NOT TESTED — Metal unavailable |
| `LiquidAI/LFM2-2.6B` | 2.6B | LFM2 hybrid | config/tokenizer only | checkpoint weights incomplete | NOT TESTED — incomplete local checkpoint |

`LFM2-350M-4bit` has 4-bit model weights. Its cache topology is mixed, so block KV and automatic
prefix restore are intentionally **UNSUPPORTED**. Dense hybrid generation remains the oracle once
the GPU qualification is run. This is a topology-safety pass, not a fallback.

## Native run results

### Qwen dense, parallel 1, APC disabled

The following are one measured deterministic-token request per context (`max_tokens=1`), after an
earlier interrupted stress run that completed 91 requests without a cache ownership failure.

| Input tokens | Status | TTFT / total ms | Peak MLX bytes | Allocated dense cache |
| ---: | --- | ---: | ---: | ---: |
| 2,045 | PASS | 2,204 / 2,204 | 11,119,168,264 | 25,165,824 |
| 8,219 | PASS | 18,859 / 18,860 | 11,170,614,024 | 103,809,024 |
| 16,409 | PASS WITH LIMITATION | 70,732 / 70,733 | 28,212,890,376 | 204,472,320 |
| approximately 32K | FAIL | HTTP 500 | n/a | n/a |

The 16K run consumed approximately 28.2 GB peak MLX memory. The 32K request failed cleanly rather
than leaking a cache handle; post-failure `/v1/runtime` reported zero active handles and zero
reserved cache bytes. On this 32 GB host, 32K is therefore not currently a safe dense baseline for
this unquantized checkpoint.

### Qwen dense plus APC, parallel 1

An approximately 7.2K common-prefix donor and three divergent-suffix borrowers produced **four
prefix misses**, zero reused tokens, and zero avoided prefill tokens. The cold TTFT was 16,914 ms;
warm TTFTs were 17,130 ms, 17,277 ms, and 17,446 ms. This is a **FAIL** for APC qualification.

The root observation is that the dense snapshot is published after the fully rendered chat prompt.
A later prompt with a different user suffix is not a token-prefix match of that full rendered prompt,
so the current server path cannot reuse the intended common prefix. Do not qualify block APC until
the prompt-prefix publication boundary is corrected and dense APC first passes.

### Qwen block smoke test

The opt-in unquantized Qwen block backend accepted a request and returned HTTP 200 with
`cache_backend=block`, 583,680 logical cache bytes, and 778,240 allocated cache bytes. This proves
basic block execution only; token parity, shared blocks, and block APC are **NOT QUALIFIED** because
the prerequisite dense APC test failed.

### LFM block rejection

`mlx-community/LFM2-350M-4bit` loaded natively and a block-handle request deterministically raised
`CacheTopologyUnsupported: block KV is currently qualified only for dense Qwen2/Qwen2.5 text
models`. This is a **PASS** for hybrid-topology safety. LFM dense, parallel, and cancellation runs
remain pending.

## How to run the matrix

Run the matrix in this order. Start a fresh server for each model/backend/parallel configuration.

1. Qwen dense, parallel 1: establish 2K/8K/16K/32K correctness and cold baseline.
2. Qwen dense plus APC: show that repeated prefixes reduce effective prefill tokens and TTFT.
3. Qwen block plus APC: show greedy dense/block token parity and nonzero shared block ownership.
4. Qwen parallel 2/4/8: establish admission and memory-pressure behavior.
5. LFM2-350M dense: run 8K/32K/64K at parallel 1/2/4/8.
6. LFM block request: verify deterministic `UNSUPPORTED`, never a fallback.
7. Cancellation stress: run both Qwen and LFM during prefill and decode.

The highest-value gate is Qwen2.5-0.5B with a 32K input, approximately 30K shared prefix, four
requests, and block APC. A qualified warm request must demonstrate the following relationship:

| Metric | Expected evidence |
| --- | --- |
| Input tokens | approximately 32,000 |
| Prefix-match tokens | approximately 30,000 |
| Effective prefill tokens | approximately 2,000 |
| Prefix cache hit | `true` |
| Shared blocks | greater than zero |
| Warm TTFT | lower than the cold baseline under the same conditions |

If runtime reports a 30K match while effective prefill remains approximately 32K, block APC is not
qualified. Record the result as a failure and keep dense as the recommended backend.

For LFM2-350M, the central gate is a 32K dense-hybrid run at parallel 4. Verify that six attention
KV states and ten convolution states remain request-isolated, are fully accounted, survive other
request cleanup, and return to baseline after completion or cancellation. Block KV is not required
for this topology: explicit rejection is the correct safety result.

For example, on the target host with a complete unquantized Qwen2.5 checkpoint:

```bash
mlx-one serve /path/to/Qwen2.5-0.5B --cache-backend dense --cache-prompt
python benchmarks/cache_runtime_bench.py --model /path/to/Qwen2.5-0.5B \
  --tokenizer-source /path/to/Qwen2.5-0.5B --offline \
  --contexts 2048,8192,16384,32768 --parallel 1 --runs 3
```

Repeat with `--cache-backend block --kv-block-size {16,32,64,128}` only for an unquantized,
batch-size-one, text-only Qwen2/Qwen2.5 checkpoint. The output JSON contains `/v1/runtime` before
and after every matrix, per-request runtime cache metrics, p50/p95 client latency and TTFT, prefix
reuse, and admission outcomes. Run separate server invocations for dense-without-APC,
dense-with-APC, and block-with-APC.

The harness verifies that `--parallel` equals the server's active scheduler configuration. Restart
the server with each of `--parallel 1`, `2`, `4`, and `8`, then invoke the harness once per value;
this avoids reporting client fan-out as server parallelism.

For several already-running model servers, use the batch form with matching URL order:

```bash
python benchmarks/cache_runtime_bench.py \
  --models /models/qwen,/models/lfm2 \
  --base-urls http://127.0.0.1:8080,http://127.0.0.1:8081 \
  --parallel 1 --contexts 2048,8192 --runs 3
```

For LFM, run only dense workloads. Verify explicit rejection for block mode and record it as
`UNSUPPORTED`, then run independent parallel requests and cancellation stress using the HTTP server.

## Results table (pending GPU execution)

| Model | Context | Backend | Parallel | Status | TTFT p50 | Decode tok/s | Peak memory |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | 2K–32K | dense | 1/2/4/8 | NOT TESTED | — | — | — |
| Qwen2.5-0.5B | 2K–32K | block | 1 | NOT TESTED | — | — | — |
| LFM2-350M-4bit | 2K–64K | dense | 1/2/4/8 | NOT TESTED | — | — | — |
| LFM2-350M-4bit | any | block/APC | any | UNSUPPORTED | — | — | — |

After native Metal execution, rename this document to **Cache Runtime M4/32 GB Qualification —
Results**, replace all pending rows with measured evidence, and use those results to decide whether
M1–M5 are closed before beginning M6 chunked prefill.
