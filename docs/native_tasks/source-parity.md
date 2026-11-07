# Source and evidence

The repository source inventory and exact hashes are in `sources.lock.yaml`; checkpoint snapshots and measured local asset hashes are in `checkpoints.lock.yaml`. The Transformers Git commit was checked with `git cat-file`; Silero's official v6.2.2 tag resolved to the prescribed commit. Recorded source file hashes do not imply every delegated helper was fully audited.

Reproducible exporter and comparator commands are described in the per-run manifests in `local/native_tasks/runs/vit-final-20261102/` and `local/native_tasks/runs/distilbert-final-20261102/`. Each manifest records file SHA-256 values, commands and environment. Evidence files are created exclusively, not overwritten; these local directories are gitignored and are not bundled with a checkout. Results and limitations are summarized in `layer-parity.md`, `processor-parity.md` and `qualification-matrix.md`.

## M2 native traces

BERT NER real checkpoint: 26 authored cases, B1 all 520 tensors pass; B2 paired traces pass, maximum B1 logits error 2.31266e-5. Final NONE/SIMPLE task comparisons pass in both regimes, including source fast-tokenizer decoder punctuation. All task outputs and tensor traces match on the separate 100-sentence quality sample. Initial decoder mismatch is retained in raw evidence and corrected in postprocessing, without encoder changes.

CLIP: generated non-square and six real images, each 39 tensors. All six full traces pass at 2 and 10 labels with exact pixels/IDs. At 100 labels all six fail only final text hidden state (two elements, max absolute 3.16143e-4); features/logits/ranking pass. The API restricts candidates to 1–10. Fixed-resolution SigLIP2: 36 tensors on one real cat fixture; exact pixels/IDs, logits max error 2.384e-5 and sigmoid max error 1.341e-7 pass. First failure is vision_layer_11; layers 11/12 and final vision norm fail (max 6.71387e-4). Numerical cause unresolved; no task admission. Evidence: local/native_tasks/runs/m2-20261103/.

## M3 traces

Silero: all1875 chunks from the pinned60-second real test WAV pass fixed FP32 gates for PCM,STFT,magnitude,four convolution stages,probabilities,recurrent state and context. Max probability error1.75834e-6, state5.72205e-5; context/STFT exact. Source timestamps match defaults and tested threshold/padding/duration settings; chunked vs offline is exact natively, including partial tails and reset/interleaving. Ten LibriSpeech clips plus3 authored negatives have identical native/oracle segments and clip decisions; maxprob error1.84774e-6.

DETR cat fixture:27 boundaries all pass. Twelve additional real COCO images:7 all-boundary pass,5 fail. First failures: image139 decoder03;872 decoder04;1000 decoder05;136466 decoder04;39769 final logits. Largest failed-logit maxabs1.08957e-3; encoder/backbone/pixels/masks pass across all12. TF32 disabled in fresh native processes. Precise numerical cause remains unresolved; no verified source-math defect was identified or tolerance changed. Raw traces/comparisons and exact commands in m3-20261103/detr-matrix. No detection task interface admitted.

## M4 document/video gates

Real-document processor: prompt IDs/grid exact and pixel patches pass FP32. Identical dense FP16 vision weights: first failing block08 (22 elements,maxabs.078125); final merged features fail (maxabs.232421875). A diagnostic same-value FP32 replay also fails, first block13, final merged maxabs.00111389. Root arithmetic cause remains unresolved; no image-encoder math/dtype/tolerance was changed to pass. Packed Q4 decoder/source QA comparison was not continued past this prerequisite gate.

Tiny FP32 source oracle for verified video corrections: all9 comparisons pass, video merged maxabs2.38419e-7, temporal positions/deltas exact for default,.5,1,2.5-second grid intervals (including pinned source integer cast). This regression is separate from real-checkpoint qualification. Real10s Big Buck Bunny clip:8 source-equivalent selected frames and grid; native per-frame/PIL packing vs source fast video processor fails11050 pixel elements, maxabs.01500797. Full video QA remains unimplemented/admission blocked by processor and dense vision gates.

## M5 depth traces

Six pinned real RGB/grayscale/square/non-square images: pixels exact, all32-boundary whole traces fail unchanged FP32. Cat first fails normalizedtap06, coffee normalizedtap03; other source-ordered archives first fail block12, with additional normalized tap/neck/fusion failures. Encoder states and normalized side taps are separate branches; source computes encoder before normalizing taps, so a reported first block12 does not imply earlier tap branches passed. Five native/resized final maps pass; camera fails19 native-depth and12 resized-depth elements. Largest normalized-tap maxabs8.55446e-4. Source-relative resized MAE7.18207e-7–5.32688e-6, RMSE2.22406e-6–8.09406e-6; camera resized maxabs4.71830e-4. Four isolated source bilinear/bicubic probes pass. No verified mathematics defect identified, no tolerance changed, no task APIs admitted.
