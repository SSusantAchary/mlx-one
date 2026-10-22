# Text quality protocol 1.0.0 — declared before execution

Dataset: `tests/alpha_0_1/fixtures/text-quality-v1.json`, `mlx-one-alpha-text-contract-1.0.0`, using existing native-task-inputs-v1 profiles and the existing task qualification worker. Ground truth is authored independently of generated model outputs: elementary facts, arithmetic, toy deduction, explicit literal/JSON/copy instructions and documented rejection contracts. This is a bounded conformance dataset, not a statistically held-out general quality benchmark. No earlier Paris/arithmetic smoke result supplies a golden.

All 27 cases must execute five measured repetitions, after independent reference execution and one warm-up. Each case exercises both Engine.stream and Engine.generate. No answer, threshold, case, stop string or prompt may change after execution to turn failure into success; future revisions require a new dataset version and new receipts. No other model executes without separate explicit authorization.

| Category | Cases | Criterion |
|---|---|---|
| Factual QA | 3 | Exact Tokyo, Jupiter, H2O answers |
| Arithmetic | 3 | Exact 12, 54, 13 |
| Simple reasoning | 3 | Explicit count, comparison and deduction answers |
| Instruction following | 3 | Literal, lowercase and list selection; no extra content |
| Structured output | 3 | Strict JSON values and types; no markdown, duplicate keys or nonfinite values |
| Chat template | 2 | Real checkpoint template with thinking disabled; system instruction and multi-turn history |
| EOS | 2 | Exact short answer and natural stop finish, with no supplied stop sequence |
| Stop sequences | 2 | Stop marker and subsequent text suppressed; finish_reason=stop |
| Unicode | 3 | Exact accented Latin, Devanagari and CJK text; no normalization |
| Invalid input | 3 | Empty/non-string prompt and invalid token budget rejected by both APIs with declared ValueError/message |

Exact-text scoring strips only outer whitespace. JSON uses structural equality with strict types. Every case and every repetition must pass: minimum accuracy 100%, five complete repetitions, all ten categories and all 27 required case IDs. Determinism compares exact token IDs/text/finish/error records. Streaming consistency compares text, emitted-token count and finish reason against generate; exceptions must agree for invalid cases. Coverage counts measurements separately from answer accuracy; a measured failing answer stays FAIL.

Qwen3 execution uses the previously authorized pinned Q8 checkpoint, reference_dense B1, FP16 activation/KV, <=512 rendered input tokens, <=64 output tokens per request (fixture maximum 16), a 120-second worker watchdog and <=24 GiB process ceiling (existing preflight can lower it). Seven independent shards keep each worker bounded; no concurrency sweep or optimization. All shards use five repetitions. Shard receipts retain incomplete global coverage; the existing scorer aggregates all required cases only when source, checkpoint, full dataset identity and repetition counts match.

Existing numerical tolerances, long-context held-out coverage gates, lifecycle and five-repetition gates remain unchanged. The correctness validator also enforces the existing five-repetition requirement, closing its prior omission. Missing HTTP service comparison, competitive performance/reference or fairness evidence stays pending; completing this bounded quality protocol does not manufacture those fields or promote a checkpoint automatically.

The service completion deadline uses the existing authorized 120-second bound. Measured task p95 can be compared with that bound; it is not an HTTP-load qualification or a competitive speed threshold. Reference speed comparison and fairness are outside a serial B1 quality run, and remain visible in aggregate gates.
