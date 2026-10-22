# Remaining gates and limitations

Phase 0 hardware correctness and the five reference workload measurements are
HARDWARE_PENDING. Backend-free implementation/testing can proceed under the
master task's explicit hardware handoff contract. This does not promote a path.

Fused numerical and performance qualification is required before completing
Phase 1's acceptance gate. Previously failing Qwen and single-query routes stay
on reference arithmetic. Phase 2 onward must not be reported complete merely
because earlier hardware tests were skipped.
Phase 1 is still in progress: its new forced-fused diagnostics and pre-dispatch
validation are implemented, while checkpoint parity and whole-model comparative
timings are unexecuted. It is not a completed fused optimization or a measured
speedup. Compatible hybrid-layer dispatch and the additional four-model fused
track require further work; existing P0.1 coverage has not been replaced.

Checksums detect changed evidence, source, tests and checkpoints; exact test
receipts reject bare/forged PASS flags. This is local reproducibility validation,
not cryptographic authentication against an attacker rewriting the entire
trusted runner and its artifacts. Explicit local checkpoint files are bound by
their hashes and declared pinned recipe; hashes alone do not prove remote origin.

Physical Metal copy traffic, temporary allocator behavior and synchronization
cost require hardware profiling. Logical payload counters do not establish
persistent-cache performance. The dense runner still rebuilds histories.

OPT-03 direct paged INT8 parity is deferred/nonblocking for independent dense
qualification. A deferred row remains DEFERRED and cannot authorize paged runs.
No other active workstream is implicitly implemented or qualified.
