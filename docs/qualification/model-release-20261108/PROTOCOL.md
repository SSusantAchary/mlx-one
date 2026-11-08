# Qualification execution protocol — 8 November 2026

## Current support policy

The user requested MLX-LM-style admission and removal of requirements MLX-LM does not impose. The current [policy](SUPPORT_POLICY.json) requires architecture/state correctness, native real-checkpoint output, relevant interfaces, regression/distribution checks and owned-weight cleanup. Passing these permits **SUPPORTED** without a blanket experimental label. Corpus quality floors and universal/all-device reference parity are diagnostic; a real native execution or interface failure remains a blocker. [Current decisions](RELEASE_CANDIDATE.md) · [repairs and remaining diagnostic limits](RESOLUTIONS.md).

Final evidence uses `checkpoint-plan-release.json` and `release-retests.json`. LFM2 Audio selects the separately pinned Q8 profile; the Q4 attempts remain recorded. Final wheel/source identity is in `candidate-build-resolution-final.json`; `candidate-lineage-resolution.json` extends the original wheel lineage. The final package differs from inference build 8 only in help text and the evidence-bound support catalog. `regression-resolution-audit.json`, `package-audit-resolution-final.json`, `clean-wheel-resolution-final.json` and `cleanup-resolution-audit.json` record the final checks. Per-model results retain their original installed wheel identity; they do not claim all 40 were rerun on the final wheel.

## Original execution protocol and intermediate findings

The sections below preserve the stricter original protocol and its intermediate findings. Statements that a repair was still pending describe that earlier stage. They are superseded by the current decisions and resolution records linked above.

This campaign executes the [release guide](../../../mlx-one_model_release_guide.md) against **one explicitly pinned representative for each of the 40 registered native model types**. Additional checkpoint sizes and profiles named in that guide retain separate qualification requirements: for example Llama-3.2, GPT-OSS TQ3, larger Whisper variants, Moonshine Base, LFM2.5-VL 3B and TrOCR Handwritten do not inherit the selected representative's result.

The clean candidate environment contains a built wheel and task dependencies, without Torch, Transformers, MLX-LM, MLX-VLM or MLX-Audio. A worker verifies installed source is inside that environment and guards reference-library imports. Candidate builds were revised as real-checkpoint failures exposed defects. Each result records its installed RECORD digest; `candidate-lineage.json` maps these to wheel identities. The final source has its own build manifest. Reinstalling the immutable local wheels and normalizing RECORD text in the same way as `importlib.metadata.read_text()` reproduces all nine installed identities used by the probes. Raw RECORD files and their checksums are archived separately. Earlier results are historical evidence from the recorded candidate, not assertions that all 40 were rerun on the final wheel. CLIP's original published pickle is converted by the existing audited converter in a separate reference environment using `torch.load(weights_only=True)`; production accepts only its hash-checked safetensors conversion. No publisher remote code executes in production.

All executions use native MLX on the Apple M4 with 32 GiB unified memory and MLX 0.32.2. The reference-dense execution mode and bounded sample sizes are deliberate scopes, not promises about training, optimization profiles, long contexts, all model sizes or other hardware.

## Gates and sample metrics

1. **Identity:** freeze full checkpoint revisions, selected file inventories and weight SHA-256; record candidate wheel/source identity. Conversions retain their own artifact digest. The Qwen1.5 MoE representative is a separately pinned MLX Q4 artifact because the publisher's 28.63-GB weights exceed the host's recommended 22.91-GB working set before caches.
2. **Architecture/regression:** enable `MLX_ONE_RUN_MLX_TESTS=1`. Skipped optional real-checkpoint tests remain skipped; they are not evidence of checkpoint admission. The first full run reported 2,521 passed, 95 skipped and one stale 36-type inventory assertion; its corrected recheck passes. Later runs are retained separately. The final full run passes 2,555 tests, with 95 skipped; publisher-style ColBERT expansion tests are included in that final run.
3. **Real checkpoint:** use task-appropriate immutable Hugging Face samples, save outputs and failures, and record strictness limits below. Fixture IDs and hashes are in `datasets.json`; raw source samples remain local.
4. **Installed interfaces:** execute the Python task API; the later worker also invokes installed Click command handlers, authenticated FastAPI routes and SSE for text/vision where applicable. These HTTP tests use FastAPI's in-process client; they do not establish external-network deployment behavior. Any missing interface replay is explicitly pending.
5. **Decision:** a successful smoke probe is not a release approval. Mark remaining quality/reference/interface gaps explicitly, retain prior failed evidence, and leave package release-approval controls unchanged. A failed or missing required gate blocks a claim that all 40 types are ready.

| Task | Frozen sample | Check and boundary |
| --- | --- | --- |
| Text | WikiText-2 raw test; first five rows with over 120 characters | Bounded completion, finite prefill/decode, cache advance/reset. Later worker: up to 96 next-token targets per excerpt; mean NLL/PPL and a PPL <1,000 sanity floor. This is not full-corpus perplexity. GPT-OSS additionally uses its chat template and a larger response budget. |
| Vision-language | Existing pinned HF documentation cat and car images | Save descriptions; expected-category keyword sanity, image changes, early stream close and subsequent request. Two examples are not a VQA benchmark. |
| ASR | Pinned LibriSpeech dummy validation utterance `1272-128104-0003`, padded from 9.9 to 10 seconds | Save transcription, WER/CER using existing audio normalization; WER ≤0.35 is a one-clip smoke floor, not corpus accuracy. |
| Forced alignment | Same speech with its transcript | Word coverage, finite ordered timestamps within audio duration. Human word-boundary accuracy is pending; utterance transcripts alone cannot qualify alignment precision. |
| VAD | Same speech plus authored silence/state controls | Detect speech presence, reject silence, exercise finalization/reset. No human segment-boundary gold or general robustness claim. |
| Fixed-vector retrieval | First 32 STS-B test pairs at the recorded revision | Finite dimensions, batched/single agreement, cosine scores, small-sample Spearman ≥0.5. This is not an MTEB score. |
| Reranking | High-similarity pairs within those 32 STS-B rows plus an authored unrelated control | Relevant pair ranks above unrelated document; finite scores. Record exact pair count. |
| Late interaction | Same frozen retrieval sample | Require a compatible publisher bundle and a genuine token-vector/MaxSim workflow. Preserve projection, prefix, punctuation-mask and query-expansion semantics; renaming a model type is insufficient. |
| Sentiment | First 32 SST-2 validation rows | Save labels/scores; accuracy ≥0.8 is a sample sanity floor. The existing full validation result remains separate. |
| NER | First 16 CoNLL-2003 test rows, tokens joined with spaces | Save real entities and offsets. Strict entity-span precision/recall/F1 is measured using the pinned label mapping; quality acceptance still requires review. |
| CLIP | Same pinned HF cat image, three candidate labels | Save candidate-conditional scores and review top-1; preserve the 1–10-label interface bound. |
| OCR | TrOCR: recorded `CASH BILL` receipt line crop. Page models: first fixture `434` from the existing seed-20261025 Omni OCR benchmark sample, at its pinned HF dataset revision | TrOCR normalized CER ≤0.2. Page models: NFKC/whitespace-normalized CER ≤0.25, plus recorded WER. Up to 1,024 generated tokens, except Florence’s supported 512-token budget. The original invalid Florence budget is retained. One patient-intake page is a smoke check, not document/layout/corpus certification; generated text is recorded even when accuracy fails. |
| Segmentation | Pinned HF SAM2 truck fixture | Nonempty concept/point masks, dimensions, score bounds; reference-mask parity is a separate gate. |
| Video segmentation | Controlled 32-frame translated truck fixture, offline and streaming | Frame accounting and lifecycle. This is a controlled fixture, not natural-video tracking quality. A standalone temporal component needs its own admitted loader and workflow. |

The small sample floors are declared harness checks, not upstream quality guarantees. A model may execute successfully while its quality or release status remains blocked.

## Disk ownership

Only a directory created under the campaign's `models/` root, carrying its exact ownership marker, may be removed. Results are written to a temporary file, flushed with `fsync`, then atomically renamed before cleanup. Failure and timeout reports also precede deletion. Existing user checkpoints are never adopted for deletion. The preserved native GPT-OSS checkpoint is reused only after verification; the preserved GGUF folder is never a native test target. Xet chunk caching is disabled to avoid retaining downloaded weight chunks.

Separate controllers downloaded disjoint cohorts while a shared file lock serialized all native GPU workers, candidate installs and native regression runs. Early Xet transfers stalled; their handoff records are retained. Large assets subsequently used bounded 64-MiB HTTP ranges with four transfer workers, size/Content-Range checks and full-file SHA verification. Chunk files and merges stay inside the owned model container and are deleted with it; cleanup-byte totals can include duplicate transfer chunks and conversions.

The transfer handoffs retained incomplete owned downloads for explicit resumption. Its owned directory and duplicate partials are removed after the final checkpoint result, and the cleanup record includes those bytes. Resume requires an ownership match and the `--resume-owned-downloads` flag.

## Corrections and limitations

- OpenELM's recommended gated tokenizer returned 403. The public pinned tokenizer's
  SentencePiece file is byte-identical to the publisher-recommended vocabulary;
  its full ID mapping and representative encodings were checked. Provenance and
  the original access failure are retained. OpenELM now inserts its required BOS
  exactly once; native/reference logits agree within the recorded error.
- LFM2 raw completion also requires BOS. The shared direct, actor, cache-preview
  and accounting paths now use the same token IDs. Its corrected five-excerpt
  PPL is 62.26; the source FP32 versus native Q4 NLL differences are recorded,
  without claiming exact quantized/source parity.
- Qwen2 MoE's native Q4 checkpoint uses per-expert packed weight metadata. The
  sanitizer preserves packed values while assembling quantizable routed experts;
  only selected experts execute. Tiny quantized-load regressions and a real
  checkpoint retest pass.
- Mean-pooling BERT/MPNet bundles may carry unused pooler weights. Only those
  known unused tensors are dropped, and only under the validated configuration.
- Qwen retrieval tokenization admits the published offset-only ByteLevel processor
  and its composition with a special-token template. The bare embedding backbone
  weight prefix is normalized with duplicate detection. Qwen3 Embedding still
  fails the original strict batch/single tolerance; that failure is preserved.
- Fixed-vector retrieval now has native ModelManager/server integration. Explicit
  `--retrieval-task reranking` handles bundles without task metadata. HTTP services
  hold a model lease during scheduled operations. The serving integration tests
  include automatic wiring and lease retention.
- The validated ColBERT module/projection/prefix/expansion/punctuation recipe has
  a public token-feature/MaxSim workflow. Path registration loads lazy weights on
  the device owner; a multi-sequence request requires an explicit sequence budget.
  The initial stock-reference comparison exposed an expansion convolution mismatch
  (maximum query-vector error 0.1133). Matching the publisher's unmasked expansion
  convolution reduces query/document error below 3.7e-7. The final comparison uses
  the stock publisher AutoModel forward, independently prepared token IDs/masks
  and published Dense projection; no reference workaround is used.
- LFM2 Audio retains passing ASR/TTS/short-STS outputs while its longer STS case
  fails the existing 512-frame generation guard. The guard was not raised.
- The standalone temporal SAM tracker has no admitted public checkpoint loader.
  Composite SAM video output cannot be counted as proof of this component.

No registry support enum, website support claim, qualification decision or
publication approval is promoted by the campaign. Failed mandatory gates remain
blocked. Additional profile sizes, quantizations, optimization modes, contexts,
datasets and hardware in the release guide are not implied by one representative.

## Replay

Use an installed environment with `huggingface-hub` to run the controller and a separate clean candidate-wheel environment with `[server,vision,ocr,ocr-bench]` plus SciPy to run workers. Set the following paths to new task-owned locations:

```bash
WORK=/private/tmp/mlx-one-release-replay
EVIDENCE=docs/qualification/model-release-replay
PYTHON=/path/to/controller/python
CANDIDATE=/path/to/clean-wheel/bin/python
REFERENCE=/path/to/isolated-reference/bin/python
# OpenELM source reference needs Transformers 4.40.2.
# ColBERT/LFM2 reference probes use the separate Transformers 4.57.1 environment.

# The checked-in checkpoint-plan.json is the immutable inventory for this run.
# Run the inventory command only when intentionally selecting a NEW cohort.
# "$PYTHON" scripts/qualification/model_release_inventory.py --evidence "$EVIDENCE"

mkdir -p "$EVIDENCE"
cp docs/qualification/model-release-20261108/checkpoint-plan.json "$EVIDENCE/checkpoint-plan.json"
"$PYTHON" scripts/qualification/model_release_fixtures.py --work "$WORK" --evidence "$EVIDENCE"
"$PYTHON" scripts/qualification/model_release_campaign.py \
  --plan "$EVIDENCE/checkpoint-plan.json" --work "$WORK" --evidence "$EVIDENCE" \
  --python "$CANDIDATE" --reference-python "$REFERENCE"
"$PYTHON" scripts/qualification/model_release_report.py --evidence "$EVIDENCE"
```

A completed result is immutable for a run: use a new evidence directory to retest a model after changing code or fixtures. Keep the original failure visible. `--models` selects a subset; it does not count unselected types as tested. The campaign never publishes a package or pushes Git changes.


## Evidence audit

All 40 selected final results have archived worker source snapshots and completed
cleanup records. Five early pilot reports retain worker digests and wheel
identities, but their original worker source bytes were not archived. Each of
those five has an explicit later replay with an archived worker snapshot; those
replays are selected in the final report. The gap remains visible in
[evidence-audit.json](evidence-audit.json).
