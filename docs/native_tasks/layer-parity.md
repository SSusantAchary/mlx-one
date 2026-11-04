# Layer parity

The comparator uses the existing `inference_baseline.TOLERANCES.float32` gate (`atol=1e-4`, `rtol=1e-4`) without modification. It verifies tensor inventory, shape, finite values and every element. Top-1 agreement is supplementary and never overrides elementwise failure.

T01: on one synthetic non-square image, input and logits pass; the first divergent intermediate layer is 05. Layers 05, 06, 10, 11 and 12 have elements outside the gate. Final normalized hidden state and logits pass. This remains a failed whole-trace gate. An earlier exploratory reference trace applied layer normalization twice and is superseded by the explicit forward-hook run; both remain in ignored local evidence.

T02: all tokenizer, embedding, six layer, logits and probability comparisons pass in 21 individual cases, including truncation. This is an input-regime-specific numerical result, not a public task qualification. Raw machine-readable outputs and manifests are under ignored `local/native_tasks/runs/`.

The follow-up investigation classified the one-fixture discrepancy as accumulated backend numerical execution differences, with the precise kernel-level cause unresolved. Identical-input individual operations in blocks 4 and 5 pass, but the full sequence remains outside the unchanged FP32 gate. See [the first-divergence report](vit-first-divergence.md).

M1 source revalidation: the preserved original 21-case HF trace compares successfully with the task-adapted native code (231 tensors). Each expanded 27-case B2 trace contains 154 compared tensors and passes at both max lengths 512 and 32; maximum logits error is `2.3366e-5`. This extends the tested batch/input regime without altering the encoder equations or global numerical gates.

M2–M5 layer/operator reports are summarized in source-parity.md and phase-reports/M2.md–M5.md. Raw comparator JSONs record every tensor, max/RMS/relative errors and elementwise failures. NER and admitted CLIP/VAD profiles pass their tested traces. SigLIP2, DETR, Qwen document/video and depth gates remain failed; their passing final outputs or small regression probes do not override those failures.
