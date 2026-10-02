# Cache Runtime Qualification

Record measured results only. Keep dense as the recommended backend until block parity, lifecycle,
and performance gates pass on the target model and machine.

## Test commands

```bash
.venv/bin/pytest -q tests/test_cache_runtime.py tests/test_engine_contracts.py \
  tests/test_core_cache_capacity.py tests/test_server_scheduler.py
.venv/bin/pytest -q
.venv/bin/ruff check .
```

## M4 32 GB matrix

Run a supported small (0.5B–1B), medium (~3B), and larger (7B–9B) Qwen2/Qwen2.5 checkpoint where
available. Test 1K, 2K, 8K, 16K, 32K, 64K, and 128K only when the checkpoint supports the context.

For each model/context record:

| Model | Context | Backend | TTFT ms | Prefill tok/s | Decode tok/s | Peak bytes | KV bytes | Allocation ms | Output digest |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| | | dense | | | | | | | |
| | | block | | | | | | | |

## Repeated-prefix workload

Use an 8K shared prefix, eight requests, a 256-token unique suffix, and 128 generated tokens. Compare
prefix disabled, dense snapshot, and block APC. Record executed/avoided prefill tokens, p50/p95 TTFT,
peak memory, cache bytes, hit rate, and total wall time.

## Agent-session workload

Use an approximately 8K system prompt, 4–6K tool definitions, and 5–10 growing follow-up turns.
Record cold/warm TTFT, total prefill tokens, prefix hit rate, cache memory, and total wall time.

## Acceptance record

- Dense, block, and block-prefix greedy token IDs are identical.
- Cancellation and allocator stress tests leave no handles, reservations, references, or pins.
- Existing server and streaming behavior passes the full suite.
- B=1 block decode regression is measured. If it exceeds approximately 5%, document it and retain
  dense as the recommended backend.
