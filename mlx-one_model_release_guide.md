# MLX-One model release guide

Prepared: **8 November 2026**. MLX-One baseline: [`fd44416`](https://github.com/SSusantAchary/mlx-one/tree/fd444163597472668b78076308d51e5499ccba66).

**0.1.1 approved for release on 8 November 2026. Publication is pending.**
See the [release notes](docs/releases/0.1.1.md) and
[maintainer decision](docs/releases/0.1.1.json).

This guide covers all **40 registered native model types**, adapting MLX-LM admission practice to each modality. After execution and repairs, all 40 have passing real-checkpoint primary workflows. The [current model support report](docs/qualification/model-release-20261108/RELEASE_CANDIDATE.md) binds these claims to pinned checkpoints, task outputs, applicable interfaces, regression tests and distribution checks. No artifact has been published.

**Current policy, following the user's instruction:** use **SUPPORTED** when implementation, real checkpoint execution and relevant interface checks pass. There is no automatic **Experimental** label and no additional broad-reliability certification requirement. MLX-LM requires model tests and actual generation output; its template makes reference output conditional on CUDA availability. See [contribution instructions][LM-CONTRIB], [new-model template][LM-TEMPLATE] and [CI][LM-CI].

Corpus CER/WER/F1/PPL floors, universal numerical tolerances, all-device reference parity and optional task extensions are diagnostics or checks for specific advertised claims. They are not universal model-support gates. The stricter initial campaign, its failed measurements and the pre-campaign 23/7/10 audit remain historical evidence; they are not the current support count. [Policy](docs/qualification/model-release-20261108/SUPPORT_POLICY.json) · [repairs and diagnostic limits](docs/qualification/model-release-20261108/RESOLUTIONS.md) · [all 40 models](status_models_testing.md).

## 1. Release objective and scope

Use the practical MLX-LM-style standard: **architecture tests + exact real-checkpoint functional verification + affected regression checks + reproducible installation and documented limits**. For non-text models, replace text generation with the actual task: transcription, waveform generation, embeddings, reranking, masks, classifications or entities.

A supported release claim is a tuple:

`model type + checkpoint ID + immutable revision + weight/conversion identity + precision + task + interface + hardware/software + tested input bounds`.

Do not require an exhaustive benchmark leaderboard, every model size, every device or every intermediate activation before documenting support. Additional quality, numerical, long-context, streaming, training and performance claims need evidence appropriate to those claims. Real loading, execution, invalid-output and interface defects remain blockers. Historical accuracy failures remain recorded as failures even when the current support policy does not make that benchmark a release gate.

The 40 types are not 40 separate families or 40 test cases. For example, SAM3 has four registered components; `gpt_oss` has separate MXFP4 and TQ3 targets; LFM2 Audio has ASR, TTS and STS tasks. The five dashboard catalogs contain 49 checkpoint/task/precision profiles and are a different denominator.

## 2. What the upstream projects actually establish

These are observed public practices, not a shared certification standard. Former `Blaizzy/mlx-vlm` and `Blaizzy/mlx-audio` links resolved to `Neywa-Labs` at the inspected revisions.

| Source | Observed practice | What MLX-One should adopt |
| --- | --- | --- |
| **MLX-LM** | Contributor instructions require new model tests; the new-model template asks for a real generation output and a reference output if CUDA is available. PR/main CI runs unit and parallel tests. | Tiny architecture/cache tests plus saved real-checkpoint output and shared runtime regressions. Reference comparisons should be available where practical; do not invent a universal exact-logit threshold. |
| **MLX-VLM** | JSON small-model cases drive model/processor tests. Contributor guidance explicitly requires idempotent sanitization. CI separates interface, model/processor, training/cache and media suites. | Test image/media processing and conversion contracts independently from generation; use recorded media fixtures. |
| **MLX-Audio** | Its adding-model guide demonstrates load-and-generate with a nonempty waveform. CI separates core, modular-import, vendored parity and task suites. There are WER-runner tests. | Use task-specific input/output contracts, waveform or transcript fixtures, streaming state tests and modality-specific regressions. |
| **MLX-Embeddings** | Small-config model/pooling/convert tests exist. The ordinary PR workflow explicitly ignores `test_smoke.py`; that script loads real models and exercises embeddings. | Keep fast tensor/pooling tests and a separate real-checkpoint embedding/retrieval smoke; a green unit job alone is not real-weight proof. |
| **Hugging Face** | Hub docs describe model metadata/cards, versioned downloads, uploads and reporting; Transformers distinguishes unit/processor/model testing from integration behavior. | Pin identities and processing assets, document intended task and limits, record reference behavior and publish reproducible usage. Uploading weights is a publishing operation, not proof of task quality. |

Sources: [MLX-LM contribution guide][LM-CONTRIB], [new-model template][LM-TEMPLATE], [LM CI][LM-CI]; [VLM contribution guide][VLM-CONTRIB], [VLM model cases][VLM-CASES], [VLM CI][VLM-CI]; [Audio adding a model][AUDIO-ADD], [Audio CI][AUDIO-CI]; [Embeddings tests][EMBED-TESTS], [Embeddings CI][EMBED-CI], [manual smoke][EMBED-SMOKE]; [HF writing model tests][HF-TESTS], [model cards][HF-CARD], [downloads][HF-DOWNLOAD], [uploads][HF-UPLOAD].

## 3. Shared release gates

Every model-specific entry in section 7 uses the engineering checks below. Accuracy/reference checks in the original detailed plan are additional diagnostics unless the release advertises that specific accuracy, task extension or numerical claim. The current decisions and exact checkpoints are in the model support report; the per-model starting statuses and next actions below preserve the original plan.

| Gate | Required evidence | Completion rule |
| --- | --- | --- |
| **G0 — Identify the scope** | Model/task/interface, checkpoint ID, full revision, precision, config/tokenizer/processor hashes, conversion recipe, source commit and intended machine. | No moving `main` or filename-only identity; known unsupported task/profile stays excluded. |
| **G1 — Architecture and weight contracts** | Tiny real native forward, dtype/shape/mask/state checks, strict weight mapping, conversion tests and invalid-config rejection. Test quantized layouts that will be advertised. | All required assertions pass; a skip is not a pass. Save/load or sanitize twice must not corrupt a converted bundle. |
| **G2 — Real checkpoint task smoke** | At least one actual pinned checkpoint per advertised model/profile, with saved real input and output plus task-specific checks below. Record load success, completion, actual sizes and measured memory. | The model does the named task through native code. A tiny/random model or merely existing source file is insufficient. |
| **G3 — Public interface and state** | Test each advertised Python/CLI/HTTP/streaming surface, task bounds, invalid inputs, stop/finalize/reset and affected cleanup/cancellation behavior. | Public behavior matches the documented scope. Do not require a nonexistent HTTP route merely because another model has one. |
| **G4 — Regression and distribution** | Relevant architecture/processor/cache/runtime suite, existing release checks, distribution audit and clean-wheel execution outside the source checkout. | Native execution is actually exercised on Metal where claimed; installed code/assets are the tested artifact; skipped hardware/weight tests remain explicit gaps. |
| **G5 — Publish the evidence and review** | Profile record, test commands/logs, artifact hashes, examples, limitations, licenses/notices and reviewer decision; website/manifests reflect the same scope. | Scoped support can be advertised after applicable gates pass and review is complete. Package publication and optional weight publication remain separate actions. |

### Claim-dependent checks

- **Reference behavior:** compare a small fixture with the publisher/reference implementation where practical, in an isolated environment. Record exact reference revision, preprocessing and dtype. This is diagnostic unless making a specific parity claim. Cross-runtime generated text need not be bit-for-bit identical. Preserve the measured outcome and original tolerance.
- **Task quality:** save diagnostic metrics appropriate to the task. Set broader corpus/accuracy acceptance criteria before executing a quality claim; no universal accuracy percentage is invented here. A small positive fixture proves functionality, not general accuracy. Preserve failed/truncated examples.
- **Performance:** when changing performance-sensitive code or publishing a speed claim, compare the same workload/precision/outputs/hardware before and after. Keep direct-model, full task pipeline, Python Engine and HTTP SSE timing scopes separate. Pure support admission does not imply a throughput promise.
- **Extensions:** full-context, multi-image, video, multilingual, streaming, batch, quantization, tool use, training and model merging are separate checked scopes. Only advertise an extension whose relevant tests passed.
- **Reference dependencies:** MLX-LM/VLM/Audio/Embeddings, PyTorch and Transformers may be isolated validation references. Keep MLX-One's native execution boundary and current wheel import guard; adding a reference dependency to production just to pass a test is not the release plan.

## 4. Task-specific acceptance packs

| Pack | Real smoke and core assertions | Metrics to retain / scope limitations |
| --- | --- | --- |
| **T — Text** | A bounded prompt and checkpoint chat template where relevant; prefill, decode, EOS, seeded generation, continuation/reset, quantized load and advertised serving/streaming. | Actual prompt/output counts, output text, load/latency/memory; no blanket size/context claim. MLX-LM's 1,024-token example is a reference practice, not a requirement to force that output on every model. |
| **V — Vision-language** | Known image with a factual question; correct processor/grid/tokens/positions, single image and only claimed additional media paths; continuation and advertised interfaces. | Save image hash, image dimensions, processed tokens and answer; generic image chat is separate from document/video QA. |
| **ASR** | Speech, silence and short-input handling; correct sample rate/frontend, actual duration, transcript, task/language options and advertised decoding path. | WER/CER with fixed normalization, language and dataset identity; timestamps/streaming/translation need distinct evidence. |
| **ALIGN** | Known words with independently timed boundaries; ordered timestamps inside original audio, unmatched-word reporting and duration accounting. | Word-boundary error distribution and declared gate. Audio-to-word alignment is not transcription; execution completion alone is insufficient. |
| **SPEECH — TTS/STS** | Finite nonempty waveform, correct sample rate, bounded duration/amplitude, intelligibility/listening notes, ordered chunks/finalization where claimed. Test STS separately with real spoken input. | Actual samples/duration, TTFA/RTF/memory; an ASR round-trip is diagnostic, not a MOS/listening certificate. |
| **VAD** | Speech/silence, frame/context/recurrent state, timestamp bounds, finalize/reset and independent sessions. | Frame probabilities and a clearly defined clip/segment metric; clip-presence success does not establish annotated boundary quality. |
| **OCR** | Readable real line/page appropriate to the model; correct processor/task prompt, output text, bounded generation and input rejection. | CER/WER normalization, failures/truncations and actual tokens. Line crops are not full pages; layout, tables and formulas are separate claims. |
| **EMBED** | Finite embedding dimensions, masks/pooling/normalization, batching consistency and saved similar/dissimilar/query-document examples. | Cosine/ranking diagnostic. Full MTEB is optional unless claiming its score; no conversion from tensor shape into retrieval-quality proof. |
| **RERANK / COLBERT** | Correct pair-scoring template or token-vector/MaxSim contract, padding exclusion, score direction and stable rankings. | Pair ordering/retrieval metric on a pinned fixture; scores or token vectors must not be mislabeled as sentence embeddings. |
| **SEG / VIDEO** | Real masks/boxes and prompt modes; image transforms, frame accounting, temporal IDs and session lifecycle for the claimed component. | Named reference comparison, mask/box scores and ID handling; natural multi-object video needs its own fixture set. |
| **CLASSIFY / NER / CLIP** | Label mapping, attention masks, valid input bounds; class probabilities, entity spans/aggregation, or candidate-prompt image/text matching as appropriate. | Accuracy, strict-span F1 or top-1 agreement with dataset/scope; softmax values are not automatically calibrated confidence. |

These packs adapt the upstream patterns to MLX-One. They do not require every model to implement every pack or interface.

## 5. Engineering gaps identified before the campaign

These findings come from this checkout; they are work items, not changes made by this guide.

1. **Stale registry count:** [wheel_smoke.py](scripts/release/wheel_smoke.py) currently asserts `len(registered_model_types()) == 36`, while the native registry has **40** types. Replace the stale assertion with a versioned expected registry identity/set derived from the release candidate and check the installed wheel against it. Do not simply remove the registry audit.
2. **Text-only wheel loop:** [.github/workflows/release.yml](.github/workflows/release.yml) iterates M01–M10 and uses the text `Engine` runner. It does not establish a 40-type multimodal release, and M05 has a separate unqualified converted-primary profile. Add task-appropriate installed-artifact runners for selected admitted profiles; explicitly exclude or fix profiles that have not passed their gates.
3. **Manifest/evidence scope:** [validate.py](scripts/release/validate.py) validates a 13-checkpoint text manifest and checks administrative approval for `alpha_supported`. Retain that protection and add reviewed modality-aware evidence validation before claiming broad support. Do not reinterpret existing text IDs as audio or image tasks.
4. **Retain approval and artifact identity:** the existing workflow already has native-hardware, distribution audit, candidate validation and a manual publication environment. Preserve those controls. A lighter upstream publishing workflow is not authorization to bypass this repository's controls.
5. **Historical fixtures and deleted weights:** several local checkpoint files were removed during storage cleanup. Restore exact required assets deliberately before execution, verify hashes, and avoid downloading all 40 types at once. Existing evidence is reusable only when its source/weight/processor identity and claimed scope still match; rerun affected checks after changes.
6. **Persist public evidence references:** archive immutable, publishable summaries and hashes without shipping raw private fixtures, local paths, model weights or temporary archives. A temporary audit report alone is not a release artifact.

## 6. Release order

| Wave | Scope | Required next work |
| --- | --- | --- |
| **R0 — Common release plumbing** | Shared validators, wheel identity and task-specific runners | Resolve the stale 36-type assertion and text-only assumptions; preserve the existing manual publication workflow. |
| **R1 — Review existing S evidence** | 23 types with bounded functional/fixture evidence | Verify proof identities, replay affected tests and selected real checkpoints, run installed artifact checks, then document exactly which profiles can ship. This is a review queue, not automatic approval of all 23. |
| **R2 — First real checkpoints for A** | 10 architecture/pending types | Start with small retrieval/OpenELM checkpoints; separately preflight large MoE targets and the standalone tracker. Complete first real-weight task smoke and public integration. |
| **R3 — Resolve P limits** | 6 OCR types and ForcedAligner | Establish positive functional outputs and honest OCR task limits; retain measured quality failures. Resolve ForcedAligner's existing timestamp gate before advertising alignment support. |
| **R4 — Broader claims** | Context, precision, streaming, languages, natural video, training and corpus quality | Schedule only for the capabilities you intend to advertise, with fixed evidence protocols and known limits. |

Priority is based on evidence availability and recovery cost, not a promise that every type will pass. Work can proceed by independent profile; there is no need to block a bounded text release on unrelated video research.

## 7. Per-model release plan — all 40 registered types

The starting statuses, proposed extra benchmarks and next actions below were written before execution. Use [current per-model decisions](status_models_testing.md) and [repair results](docs/qualification/model-release-20261108/RESOLUTIONS.md) for completion status. Universal quality/reference gates proposed in this initial plan have been superseded by the explicit support policy above.

Every entry inherits G0–G5 and its task pack. **S/P/A is the starting audit state; none of the plans below is marked completed by this document.** Existing source/test entry points identify reusable work; a README pointer is not itself an executable test or proof that the missing integration test exists.

### Text generation

#### 01. `gemma3_text`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use M09: google/gemma-3-1b-it, pinned primary profile.
- **Model-specific work:**
  - Check the intended norm/scaling convention, tied embedding/head behavior and sliding/full attention schedule.
  - Cross the tested sliding boundary, continue a cached prompt, reset and record practical output without silently changing precision.
- **Allowed claim after passing review:** Gemma3 1B text in the tested functional scope; historical numerical and answer-quality limitations remain disclosed.
- **Next action:** Replay ordinary functional smoke and regressions. Do not replace existing functional admission policy with an exhaustive all-layer research requirement.
- **Existing implementation/test entry points:** [test_new_language_contracts.py](tests/models/test_new_language_contracts.py), [test_new_language_native.py](tests/models/test_new_language_native.py), [test_alpha_cache_semantics.py](tests/models/test_alpha_cache_semantics.py).

#### 02. `gpt2`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use M01: openai-community/gpt2, stored FP32.
- **Model-specific work:**
  - Verify learned positions, tied head/tokenizer vocabulary and causal masking on a tiny configuration.
  - Replay completion, greedy/seeded sampling, cached continuation, EOS and reset; check actual generated token counts.
- **Allowed claim after passing review:** GPT-2 124M, pinned stored FP32, the replayed Python/CLI/HTTP/SSE paths and tested context only.
- **Next action:** Reconcile existing M01 proof with the release commit; replay affected generation/interface tests and clean-wheel smoke.
- **Existing implementation/test entry points:** [test_native_gpt2_architecture.py](tests/test_native_gpt2_architecture.py), [test_native_gpt2_mlx.py](tests/test_native_gpt2_mlx.py).

#### 03. `gpt_oss`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Test M12 mlx-community/gpt-oss-20b-MXFP4-Q4 and M13 manjunathshiva/gpt-oss-20b-tq3 independently.
- **Model-specific work:**
  - Test MXFP4/mixed affine and packed TQ3 arithmetic/contracts separately; never reinterpret MXFP4 as ordinary affine Q4.
  - Run each exact pinned target through prefill/decode, stop/template/tool formatting where claimed, cache/reset and advertised serving interfaces.
  - Verify a preserved local safetensors bundle against the selected revision/hashes before reuse. A similarly named GGUF file is not a substitute for the native safetensors target.
- **Allowed claim after passing review:** Separate release records for M12 and M13. The older unpinned GPT-OSS HTTP report and tiny/kernel tests do not qualify either full target.
- **Next action:** Both targets need pinned real-weight smoke. Start with a small context and record total/active size and measured memory; no assumed TQ3 throughput promise.
- **Existing implementation/test entry points:** [test_large_moe_contracts.py](tests/models/test_large_moe_contracts.py), [test_large_moe_native.py](tests/models/test_large_moe_native.py), [test_turboquant_contracts.py](tests/models/test_turboquant_contracts.py), [test_turboquant_native.py](tests/models/test_turboquant_native.py), [test_gpt_oss_tools_native.py](tests/models/test_gpt_oss_tools_native.py).

#### 04. `lfm2`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use M04: LiquidAI/LFM2-350M, pinned primary affine-Q4 profile.
- **Model-specific work:**
  - Exercise alternating convolution/attention layers, convolution carry and attention KV state.
  - Check chunked prefill, decode continuation, reset/isolation and explicit rejection of unsupported cache modes.
- **Allowed claim after passing review:** The tested dense hybrid checkpoint/profile, not all LFM2 or LFM2.5 variants.
- **Next action:** Replay primary-profile generation and hybrid-state regression checks; keep existing safe rejection behavior.
- **Existing implementation/test entry points:** [test_native_lfm_architectures.py](tests/test_native_lfm_architectures.py), [test_native_lfm_mlx.py](tests/test_native_lfm_mlx.py).

#### 05. `lfm2_moe`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use M11: LiquidAI/LFM2.5-24B-A2B-MLX-4bit and its pinned mixed profile.
- **Model-specific work:**
  - Check shared/routed experts, mixed quantization, embeddings/router dtypes and hybrid carry state.
  - Preflight total resident weight plus KV/carry memory; run bounded real-weight generation and continuation without silently substituting a smaller dense model.
- **Allowed claim after passing review:** The exact mixed-profile 24B target on the measured hardware and tested context.
- **Next action:** Current manifest records architecture-only. First full-checkpoint functional smoke and memory evidence are required.
- **Existing implementation/test entry points:** [test_native_lfm_architectures.py](tests/test_native_lfm_architectures.py), [test_native_lfm_mlx.py](tests/test_native_lfm_mlx.py).

#### 06. `llama`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use separate M06 SmolLM2-1.7B and M07 Llama-3.2-1B records. MiniCPM5 requires its own existing integration target.
- **Model-specific work:**
  - Exercise GQA, explicit head dimensions, RoPE and the selected quantized weight layout.
  - Check each checkpoint chat template, all configured EOS IDs and multi-EOS termination; never inherit results from a different Llama-compatible checkpoint.
- **Allowed claim after passing review:** Only the individually replayed Llama/SmolLM2/MiniCPM5 revisions and precisions; training quality is a separate claim.
- **Next action:** Reuse matching functional proof where valid; run the selected checkpoint from the built wheel. Retain failed training-quality results.
- **Existing implementation/test entry points:** [test_native_llama_architecture.py](tests/test_native_llama_architecture.py), [test_native_llama_mlx.py](tests/test_native_llama_mlx.py), [test_native_minicpm5_integration.py](tests/test_native_minicpm5_integration.py).

#### 07. `openelm`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Candidate from the family guide: apple/OpenELM-270M. Resolve and record a full immutable revision before acquiring weights.
- **Model-specific work:**
  - Preserve nonuniform layer widths/head counts and tokenizer/vocabulary compatibility.
  - Perform the first real-checkpoint native load, completion, cached continuation and reset; chat support needs an instruction checkpoint/template.
- **Allowed claim after passing review:** The first successfully tested exact OpenELM checkpoint for completion; architecture tests alone keep it pending.
- **Next action:** Select/pin a compatible bundle, preflight memory, then run the first real-weight smoke and install replay.
- **Existing implementation/test entry points:** [test_native_openelm_architecture.py](tests/test_native_openelm_architecture.py), [test_native_openelm_mlx.py](tests/test_native_openelm_mlx.py).

#### 08. `phi3`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use M10: microsoft/Phi-3-mini-4k-instruct, pinned primary profile.
- **Model-specific work:**
  - Verify fused projection sanitization, rotary configuration, window masks and model dtype.
  - Generate from raw/chat prompts; test EOS, cache continuation and rejected out-of-scope context/profile requests.
- **Allowed claim after passing review:** The pinned Mini-4K checkpoint and tested precision/context; no automatic larger-context qualification.
- **Next action:** Replay M10 functional and wheel evidence; list unresolved precision/context regimes explicitly.
- **Existing implementation/test entry points:** [test_new_language_contracts.py](tests/models/test_new_language_contracts.py), [test_new_language_native.py](tests/models/test_new_language_native.py), [test_alpha_cache_semantics.py](tests/models/test_alpha_cache_semantics.py).

#### 09. `qwen2`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use M02 Qwen2.5-0.5B-Instruct and M08 Qwen2.5-1.5B-Instruct as separate profile records.
- **Model-specific work:**
  - Check GQA/RoPE, quantized loading, chat formatting, EOS, prefill-to-decode and cache reset.
  - Replay batch/streaming and public serving paths that are actually advertised; block/paged caches need their own enabled-profile evidence.
- **Allowed claim after passing review:** Pinned Qwen2.5 revisions and primary profiles in reference execution; extra cache and training modes remain separate.
- **Next action:** Replay the primary profile and wheel/API smoke; do not use Coder LoRA timing as inference or held-out quality qualification.
- **Existing implementation/test entry points:** [test_native_qwen_architectures.py](tests/test_native_qwen_architectures.py), [test_native_qwen_mlx.py](tests/test_native_qwen_mlx.py).

#### 10. `qwen2_moe`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Candidate family reference: Qwen/Qwen1.5-MoE-A2.7B. Select a compatible native layout/precision and pin it; it is not an existing qualified target.
- **Model-specific work:**
  - Test routed/shared experts, top-k dispatch, router tensor shapes and sanitizer behavior.
  - Run first real-checkpoint prefill/decode, cache continuation and serving under a measured memory budget; never relabel a dense Qwen2 checkpoint as MoE.
- **Allowed claim after passing review:** Only the selected MoE checkpoint/layout after real execution. Optional dense-Qwen cache optimizations are not inherited.
- **Next action:** Resolve the checkpoint/layout and add a real-weight integration fixture before changing architecture-only status.
- **Existing implementation/test entry points:** [test_native_qwen_architectures.py](tests/test_native_qwen_architectures.py), [test_native_qwen_mlx.py](tests/test_native_qwen_mlx.py).

#### 11. `qwen3`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** T + G0–G5.
- **First release target:** Use M03: Qwen/Qwen3-0.6B, its pinned primary affine-Q8 profile.
- **Model-specific work:**
  - Verify Q/K normalization, head dimensions, RoPE and floating/quantized weight contracts.
  - Record checkpoint-specific thinking/chat-template behavior, stop IDs, cached generation and streaming.
- **Allowed claim after passing review:** Only the pinned 0.6B profile and tested prompt/context/interface scope.
- **Next action:** Revalidate M03 serving/reset evidence and clean-wheel execution; qualify new sizes and profiles separately.
- **Existing implementation/test entry points:** [test_native_qwen_architectures.py](tests/test_native_qwen_architectures.py), [test_native_qwen_mlx.py](tests/test_native_qwen_mlx.py).

### Vision-language

#### 12. `lfm2_vl`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** V + G0–G5.
- **First release target:** V04 mlx-community/LFM2-VL-450M-8bit and V05 LiquidAI/LFM2.5-VL-3B-MLX-4bit.
- **Model-specific work:**
  - Test NaFlex patch handling, pixel unshuffle/projector and convolution/attention state.
  - Use portrait/landscape images and each checkpoint’s processor; replay image chat and continuation for Q8 and Q4 as distinct records.
- **Allowed claim after passing review:** Only the two individually tested static-image profiles; no video or wider-resolution guarantee.
- **Next action:** Revalidate V04/V05 asset/processor identities and exact precision before wheel replay.
- **Existing implementation/test entry points:** [test_native_lfm_architectures.py](tests/test_native_lfm_architectures.py), [test_native_lfm_mlx.py](tests/test_native_lfm_mlx.py).

#### 13. `qwen2_5_vl`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** V + G0–G5.
- **First release target:** V02: mlx-community/Qwen2.5-VL-3B-Instruct-4bit.
- **Model-specific work:**
  - Test window/full vision attention, patch merger, visual positions and checkpoint conversion idempotency.
  - Replay static-image chat and continuation; malformed image inputs must fail cleanly.
  - Dedicated document/video QA remains blocked until its recorded task gates and public interfaces are actually satisfied.
- **Allowed claim after passing review:** Existing static-image chat only. Do not advertise document-QA or video-QA support from this pass.
- **Next action:** Keep V02 image smoke separate from native_tasks.json BLOCKED profiles. Fix affected dedicated-task failures before widening the claim.
- **Existing implementation/test entry points:** [test_native_qwen_multimodal_architectures.py](tests/test_native_qwen_multimodal_architectures.py), [test_native_qwen_multimodal_mlx.py](tests/test_native_qwen_multimodal_mlx.py), [test_qwen25_tasks.py](tests/native_tasks/test_qwen25_tasks.py).

#### 14. `qwen2_vl`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** V + G0–G5.
- **First release target:** V01: mlx-community/Qwen2-VL-2B-Instruct-4bit.
- **Model-specific work:**
  - Verify resize/tiling, image normalization, visual-grid/token insertion and multimodal RoPE metadata.
  - Replay one-image and any claimed multi-image prompts, cached continuation and image-aware serving. Keep image ordering stable.
- **Allowed claim after passing review:** The pinned static-image Q4 workflow; video is a separate future scope.
- **Next action:** Replay existing V01 image smoke and processor regressions against the release artifact; keep unsupported video unavailable.
- **Existing implementation/test entry points:** [test_native_qwen_multimodal_architectures.py](tests/test_native_qwen_multimodal_architectures.py), [test_native_qwen_multimodal_mlx.py](tests/test_native_qwen_multimodal_mlx.py).

#### 15. `qwen3_5`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** V + G0–G5.
- **First release target:** V03: mlx-community/Qwen3.5-0.8B-4bit from the VLM plan; text M05 has a different checkpoint/profile identity.
- **Model-specific work:**
  - Test gated-delta/full-attention state, visual position interpolation, tied head and multimodal token placement.
  - Replay targeted Q4 image generation; exercise text-only behavior only for the exact checkpoint where claimed.
  - Test MTP/speculation only if included in the proposed release scope; otherwise leave optional flags experimental.
- **Allowed claim after passing review:** The admitted image-Q4 profile. Do not promote the separate converted-primary text-Q4 path from this result.
- **Next action:** Replay V03 and processor/cache boundaries; keep M05 and MTP evidence separate.
- **Existing implementation/test entry points:** [test_native_qwen_multimodal_architectures.py](tests/test_native_qwen_multimodal_architectures.py), [test_native_qwen_multimodal_mlx.py](tests/test_native_qwen_multimodal_mlx.py).

### Audio & speech

#### 16. `lfm2_audio`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** SPEECH + G0–G5.
- **First release target:** A10/A12/A13: mlx-community/LFM2.5-Audio-1.5B-4bit, with distinct ASR, TTS and speech-to-speech records.
- **Model-specific work:**
  - Test audio encoder, language decoder, depth/codebook ordering and waveform codec as separate components.
  - For ASR save transcript/WER/CER; for TTS verify nonempty finite waveform, expected sample rate/duration, stable amplitude and recorded listening/intelligibility observations.
  - Replay ordered streaming PCM chunks and finalization where advertised. Preserve the failed 10-second STS case; short-question success cannot replace it.
- **Allowed claim after passing review:** ASR/TTS can be reviewed within their passed recorded scope. STS remains WIP until its declared duration/amplitude/response checks pass.
- **Next action:** Prepare three independent task records; do not issue one family-wide pass.
- **Existing implementation/test entry points:** [README.md](src/mlx_one/models/audio/lfm2_audio/README.md).

#### 17. `moonshine`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** ASR + G0–G5.
- **First release target:** A04 UsefulSensors/moonshine-tiny and A05 UsefulSensors/moonshine-base.
- **Model-specific work:**
  - Test waveform convolutions, normalization, rotary layout, encoder/cross-attention and decoder cache.
  - Replay offline English clips including variable lengths; record transcript, WER/CER and completion.
  - Reject unsupported timestamp/translation/streaming flags rather than silently ignoring them.
- **Allowed claim after passing review:** Offline English transcription for the pinned Tiny/Base models.
- **Next action:** Revalidate both existing checkpoint smokes and task rejection boundaries; add missing regression tests if the family guide exposes coverage gaps.
- **Existing implementation/test entry points:** [README.md](src/mlx_one/models/audio/moonshine/README.md).

#### 18. `parakeet_tdt`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** ASR + G0–G5.
- **First release target:** A08 nvidia/parakeet-tdt-0.6b-v2 and A09 mlx-community/parakeet-tdt-0.6b-v3.
- **Model-specific work:**
  - Test FastConformer masks, predictor recurrent state, blank/duration advancement and greedy TDT decoding.
  - Audit isolated NeMo conversion, key mapping and exact converted weights; replay native v3 offline ASR independently.
  - Retain v2 parity WIP, language-conditioning and timestamp restrictions where they are unresolved.
- **Allowed claim after passing review:** Bounded native v3 export transcription; v2 source parity and wider task scope need separate approval.
- **Next action:** Replay A09 first; keep A08’s converted archive/source identity separate and fix its unresolved parity before broader claims.
- **Existing implementation/test entry points:** [README.md](src/mlx_one/models/audio/parakeet/README.md).

#### 19. `qwen3_asr`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** ASR + G0–G5.
- **First release target:** A06/A07: mlx-community/Qwen3-ASR-0.6B-8bit and mlx-community/Qwen3-ASR-1.7B-8bit; publisher BF16 records are separate.
- **Model-specific work:**
  - Test audio chunking/frontend, padding/duration accounting, language/context prompts and generated text extraction.
  - Replay Q8 offline speech and report WER/CER; compare only against matching reference identities.
  - Do not claim streaming parity or broader language qualification while those gates remain unresolved.
- **Allowed claim after passing review:** Recorded Q8 offline ASR only; BF16 anchor and streaming failures retain their own status.
- **Next action:** Separate Q8 functional release evidence from historical BF16 numerical/anchor investigations and any streaming candidate.
- **Existing implementation/test entry points:** [README.md](src/mlx_one/models/audio/qwen3_asr/README.md).

#### 20. `qwen3_forced_aligner`

- **Starting state:** P from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** ALIGN + G0–G5.
- **First release target:** A11 mlx-community/Qwen3-ForcedAligner-0.6B-8bit; publisher original revision is a separate reference.
- **Model-specific work:**
  - Feed audio plus known transcript; check token/word mapping, finite ordered timestamps and bounds against real unpadded duration.
  - Use independently timed words to report boundary errors and unmatched words; output existence alone is insufficient.
  - Retain the existing failed 80-ms gate and historical 800-ms discrepancy until genuinely resolved or an explicitly reviewed new scope is adopted.
- **Allowed claim after passing review:** Remain WIP until a declared alignment-accuracy gate passes. Never label alignment as speech recognition.
- **Next action:** Resolve boundary accuracy and preserve the failed evidence; do not promote A11 merely because its timing benchmark completed.
- **Existing implementation/test entry points:** [README.md](src/mlx_one/models/audio/qwen3_asr/README.md).

#### 21. `silero_vad`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** VAD + G0–G5.
- **First release target:** Pinned snakers4/silero-vad-16k entry in native_tasks.json / checkpoints.lock.yaml.
- **Model-specific work:**
  - Test mono 16-kHz input, 512-sample frames, 64-sample context, recurrent state and tail padding.
  - Verify speech/silence clips, reset/isolation, final timestamps bounded to original audio and the admitted PCM/WAV input contract.
  - Replay probability/state parity using the selected safetensors identity; do not silently compare with a JIT containing different weights.
- **Allowed claim after passing review:** Bounded FP32 mono-16-kHz VAD, stateful Python sessions and the documented stateless task route; no broad music/noise guarantee.
- **Next action:** Restore the selected small checkpoint if needed and replay native/API/wheel evidence; keep human-boundary quality separate.
- **Existing implementation/test entry points:** [test_silero.py](tests/native_tasks/test_silero.py).

#### 22. `whisper`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** ASR + G0–G5.
- **First release target:** A01 openai/whisper-tiny; A02 openai/whisper-large-v3-turbo; A03 distil-whisper/distil-large-v3.
- **Model-specific work:**
  - Test log-Mel/tokenizer/frontend, decoder/cache and the supported language/task flags.
  - Replay pinned speech fixtures with saved transcripts and WER/CER normalization; handle silence, short clips and invalid audio.
  - Translation, timestamps and long-form/streaming require distinct tests; do not promise them for every Whisper-derived checkpoint.
- **Allowed claim after passing review:** Checkpoint-specific transcription capabilities and tested languages; all three checkpoint/precision records remain separate.
- **Next action:** Replay existing short-form evidence and the advertised Python/CLI interfaces. Preserve version-specific decoder/tokenizer choices.
- **Existing implementation/test entry points:** [test_native_whisper_architecture.py](tests/test_native_whisper_architecture.py), [test_native_whisper_mlx.py](tests/test_native_whisper_mlx.py), [test_native_whisper_integration.py](tests/test_native_whisper_integration.py).

### OCR & documents

#### 23. `deepseek_ocr`

- **Starting state:** P from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** OCR + G0–G5.
- **First release target:** O07 mlx-community/DeepSeek-OCR-4bit.
- **Model-specific work:**
  - Test image tiling/crop strategy, visual projection, prompt format and safe weight conversion.
  - Replay direct page OCR and any advertised region/crop modes with explicit preprocessing.
  - Save text and geometry only for interfaces that implement them; report failures, truncations, CER/WER and actual tokens.
- **Allowed claim after passing review:** The tested native Q4 OCR pipeline; external remote-code behavior is not proof of a native feature.
- **Next action:** Resolve functional positive fixtures and wheel replay; preserve the 41 truncations and current corpus accuracy limitations.
- **Existing implementation/test entry points:** [test_native.py](tests/ocr/test_native.py), [test_documents.py](tests/ocr/test_documents.py), [test_interfaces.py](tests/ocr/test_interfaces.py), [test_corpus_benchmark.py](tests/ocr/test_corpus_benchmark.py).

#### 24. `florence2`

- **Starting state:** P from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** OCR + G0–G5.
- **First release target:** O04 microsoft/Florence-2-base-ft.
- **Model-specific work:**
  - Test task tokens, encoder-decoder state and postprocessed text/geometry contracts.
  - Keep transcription, detection, region and document task tokens separate; verify coordinates against image bounds when claimed.
  - Save functional positive outputs alongside failure/truncation and CER/WER records.
- **Allowed claim after passing review:** Only the task tokens and output forms actually replayed; no automatic general handwriting qualification.
- **Next action:** Use correct task prompts and compare a bounded functional fixture; preserve page truncations and poor small-sample handwriting results.
- **Existing implementation/test entry points:** [test_native.py](tests/ocr/test_native.py), [test_documents.py](tests/ocr/test_documents.py), [test_interfaces.py](tests/ocr/test_interfaces.py), [test_corpus_benchmark.py](tests/ocr/test_corpus_benchmark.py).

#### 25. `got_ocr2`

- **Starting state:** P from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** OCR + G0–G5.
- **First release target:** O03 mlx-community/GOT-OCR2_0-4bit; incompatible HF layout variants must not be renamed.
- **Model-specific work:**
  - Test selected config/weight layout, image preprocessing, task prompts and bounded decode.
  - Replay a known readable document and cropped-line case; record failures, truncation, CER/WER and actual token count.
  - Only test box/region/batch paths if they are part of the advertised release scope.
- **Allowed claim after passing review:** The tested transcription modes; no blanket high-accuracy or structured-extraction claim from the existing benchmark.
- **Next action:** Review the 97/100 page completion and 56 truncations; add functional reference output and retain poor-quality cases before scoped promotion.
- **Existing implementation/test entry points:** [test_native.py](tests/ocr/test_native.py), [test_documents.py](tests/ocr/test_documents.py), [test_interfaces.py](tests/ocr/test_interfaces.py), [test_corpus_benchmark.py](tests/ocr/test_corpus_benchmark.py).

#### 26. `lightonocr`

- **Starting state:** P from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** OCR + G0–G5.
- **First release target:** O06 lightonai/LightOnOCR-2-1B.
- **Model-specific work:**
  - Test vision/text projection, checkpoint dtype and image/text token alignment.
  - Replay a bounded page plus clean line fixture; cap memory and record completion, truncation and normalized CER/WER.
  - Compare same preprocessing and generated token budget when making reference or performance comparisons.
- **Allowed claim after passing review:** The pinned BF16 transcription profile on measured hardware; broader corpus claims need their own evidence.
- **Next action:** Add functional artifact/wheel replay to the corpus records; retain the 64 page truncations as an unresolved limitation.
- **Existing implementation/test entry points:** [test_native.py](tests/ocr/test_native.py), [test_documents.py](tests/ocr/test_documents.py), [test_interfaces.py](tests/ocr/test_interfaces.py), [test_corpus_benchmark.py](tests/ocr/test_corpus_benchmark.py).

#### 27. `paddleocr_vl`

- **Starting state:** P from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** OCR + G0–G5.
- **First release target:** O05 mlx-community/PaddleOCR-VL-1.5-4bit.
- **Model-specific work:**
  - Test visual preprocessing, language token insertion and document prompt/decode.
  - Replay direct page transcription; independently test table/formula/region output if advertised.
  - Do not attach the external layout pipeline’s quality to direct native page transcription.
- **Allowed claim after passing review:** The replayed direct OCR modes; layout accuracy and structured extraction remain separate.
- **Next action:** Reproduce positive page output, retain truncation/quality evidence and explicitly list excluded layout tasks.
- **Existing implementation/test entry points:** [test_native.py](tests/ocr/test_native.py), [test_documents.py](tests/ocr/test_documents.py), [test_interfaces.py](tests/ocr/test_interfaces.py), [test_corpus_benchmark.py](tests/ocr/test_corpus_benchmark.py).

#### 28. `trocr`

- **Starting state:** P from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** OCR + G0–G5.
- **First release target:** O01 microsoft/trocr-base-printed and O02 microsoft/trocr-base-handwritten.
- **Model-specific work:**
  - Test image encoder, autoregressive text decoder, start/stop tokens and processor normalization.
  - Use printed line crops for O01 and handwriting line crops for O02; save exact line transcripts and CER/WER with fixed normalization.
  - Verify full pages are rejected or require explicit line segmentation; do not invent a page capability from line-model throughput.
- **Allowed claim after passing review:** Separate printed-line and handwritten-line workflows; only the tested checkpoint/crop contract.
- **Next action:** Add clean deterministic positive line fixtures to the existing measurements. Keep out-of-domain printed-on-handwriting results visible.
- **Existing implementation/test entry points:** [test_native.py](tests/ocr/test_native.py), [test_documents.py](tests/ocr/test_documents.py), [test_interfaces.py](tests/ocr/test_interfaces.py), [test_corpus_benchmark.py](tests/ocr/test_corpus_benchmark.py).

### Embeddings & reranking

#### 29. `bert`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** EMBED + G0–G5.
- **First release target:** Candidate sentence-transformers/all-MiniLM-L6-v2; select and pin the exact sentence-transformer bundle.
- **Model-specific work:**
  - Verify BERT masks, mean pooling excluding padding, normalization and tokenizer/truncation configuration.
  - Check finite output shape, single-versus-batched consistency and stable cosine ordering for authored similar/dissimilar texts.
  - Compare the selected pooling recipe with the publisher reference; do not treat arbitrary BERT hidden states as a sentence embedding model.
- **Allowed claim after passing review:** The pinned MiniLM embedding recipe; broad retrieval quality needs a separate dataset evaluation.
- **Next action:** Run the first real-checkpoint embedding smoke; existing synthetic tests alone keep it architecture-only.
- **Existing implementation/test entry points:** [test_native_sentence_transformers_architectures.py](tests/test_native_sentence_transformers_architectures.py), [test_native_sentence_transformers_mlx.py](tests/test_native_sentence_transformers_mlx.py), [test_task_retrieval_runtime.py](tests/test_task_retrieval_runtime.py), [test_task_retrieval_native.py](tests/test_task_retrieval_native.py).

#### 30. `lfm2_colbert`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** COLBERT + G0–G5.
- **First release target:** Candidate LiquidAI/LFM2-ColBERT-350M; verify that the publisher bundle matches the native tokenizer/indexing contract, then pin it.
- **Model-specific work:**
  - Test rank-three token embeddings, query/document masks, hybrid backbone and masked MaxSim.
  - Verify padding is excluded and a small document ranking example works with the actual publisher input recipe.
  - Keep late-interaction outputs outside the generic rank-two single-vector embed interface.
- **Allowed claim after passing review:** Only the compatible token-embedding/MaxSim workflow. Publisher compatibility is unresolved until the real bundle passes.
- **Next action:** Resolve bundle compatibility first; do not bypass validation by renaming configuration fields.
- **Existing implementation/test entry points:** [test_native_lfm_architectures.py](tests/test_native_lfm_architectures.py), [test_native_lfm_mlx.py](tests/test_native_lfm_mlx.py), [test_task_retrieval_runtime.py](tests/test_task_retrieval_runtime.py), [test_task_retrieval_native.py](tests/test_task_retrieval_native.py).

#### 31. `mpnet`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** EMBED + G0–G5.
- **First release target:** Candidate sentence-transformers/all-mpnet-base-v2; resolve a full revision.
- **Model-specific work:**
  - Test relative positions, masks, publisher pooling and normalization.
  - Check finite dimensions, padding-invariant outputs, batching and a small saved similarity/retrieval example.
  - Use matching sentence-transformer processor/pooling assets in a reference comparison.
- **Allowed claim after passing review:** The pinned MPNet sentence-embedding checkpoint and tested input lengths.
- **Next action:** Obtain a compatible pinned bundle and record the first real embedding and installed-artifact replay.
- **Existing implementation/test entry points:** [test_native_sentence_transformers_architectures.py](tests/test_native_sentence_transformers_architectures.py), [test_native_sentence_transformers_mlx.py](tests/test_native_sentence_transformers_mlx.py), [test_task_retrieval_runtime.py](tests/test_task_retrieval_runtime.py), [test_task_retrieval_native.py](tests/test_task_retrieval_native.py).

#### 32. `qwen3_embedding`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** EMBED + G0–G5.
- **First release target:** Candidate Qwen/Qwen3-Embedding-0.6B; resolve and pin before execution.
- **Model-specific work:**
  - Test instruction formatting, last-real-token pooling, padding side and normalization.
  - Test requested embedding dimensions only where implemented; saved query/document vectors must have expected finite shapes.
  - Check batching and a small relevant/irrelevant document retrieval fixture using the task-specific model, not a Qwen chat checkpoint.
- **Allowed claim after passing review:** The tested instructed embedding/dimension contract; no MTEB or multilingual quality claim without evaluation.
- **Next action:** Complete first real-weight retrieval smoke and public interface replay.
- **Existing implementation/test entry points:** [test_native_qwen3_retrieval.py](tests/test_native_qwen3_retrieval.py), [test_native_qwen3_retrieval_mlx.py](tests/test_native_qwen3_retrieval_mlx.py), [test_task_retrieval_runtime.py](tests/test_task_retrieval_runtime.py), [test_task_retrieval_native.py](tests/test_task_retrieval_native.py).

#### 33. `qwen3_reranker`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** RERANK + G0–G5.
- **First release target:** Candidate Qwen/Qwen3-Reranker-0.6B; resolve and pin.
- **Model-specific work:**
  - Test the official query/document instruction template and yes/no token-logit scoring.
  - Check finite pair scores, probability bounds, ranking direction, stable ties and batched/single agreement.
  - Keep rerank outputs separate from embedding vectors; save relevant/irrelevant pair examples.
- **Allowed claim after passing review:** The tested pair-scoring/reranking workflow and input bounds.
- **Next action:** Add a pinned real-checkpoint rank-order/reference smoke before promoting the architecture-only entry.
- **Existing implementation/test entry points:** [test_native_qwen3_retrieval.py](tests/test_native_qwen3_retrieval.py), [test_native_qwen3_retrieval_mlx.py](tests/test_native_qwen3_retrieval_mlx.py), [test_task_retrieval_runtime.py](tests/test_task_retrieval_runtime.py), [test_task_retrieval_native.py](tests/test_task_retrieval_native.py).

### Segmentation

#### 34. `sam3`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** SEG + G0–G5.
- **First release target:** facebook/sam3 at the existing pinned revision in docs/sam3-qualification-m4-32gb.md.
- **Model-specific work:**
  - Verify detector/text/geometry weight contract and the real concept-image fixture.
  - Check nonempty/empty detections, mask/box coordinate sizes and score bounds; compare with the documented CPU reference.
  - Box exemplars and automatic masks need their own evidence if claimed; historical preprocessing changes require affected fixture recapture.
- **Allowed claim after passing review:** The recorded concept-image prompt scope; no promotion of every SAM3 prompt mode.
- **Next action:** Replay the selected concept fixture after current preprocessing; retain prior failed captures and package qualified=False until review.
- **Existing implementation/test entry points:** [test_sam3.py](tests/test_sam3.py), [test_sam3_workflows.py](tests/test_sam3_workflows.py), [test_sam3_comparison.py](tests/test_sam3_comparison.py).

#### 35. `sam3_tracker`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** SEG + G0–G5.
- **First release target:** Same pinned facebook/sam3 bundle, interactive component.
- **Model-specific work:**
  - Test a positive point and explicit negative/empty cases, image resize/coordinate mapping and multimask output.
  - Replay the corrected-preprocessing interactive fixture, including candidate mask ordering and predicted-IoU output.
  - Add box prompting and refinement only after their own functional/reference checks.
- **Allowed claim after passing review:** The tested interactive point-prompt scope; refinement, automatic-mask and other prompt modes remain excluded.
- **Next action:** Replay interactive fixture and supported public Python/CLI paths; do not inherit concept-detector confidence semantics.
- **Existing implementation/test entry points:** [test_sam3.py](tests/test_sam3.py), [test_sam3_workflows.py](tests/test_sam3_workflows.py), [test_sam3_comparison.py](tests/test_sam3_comparison.py).

#### 36. `sam3_tracker_video`

- **Starting state:** A from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** VIDEO + G0–G5.
- **First release target:** Standalone temporal tracker, currently architecture-only inside the SAM3 family.
- **Model-specific work:**
  - Test memory encoding/attention, object pointers, state updates and reset in isolation.
  - Require a standalone real-checkpoint integration route and reference fixture before claiming standalone support.
  - A passing composite sam3_video session is not a standalone tracker test.
- **Allowed claim after passing review:** Keep architecture-only unless standalone integration and its actual public contract are implemented and tested.
- **Next action:** Decide whether to ship it only as an internal component or implement/admit a dedicated standalone workflow.
- **Existing implementation/test entry points:** [test_sam3.py](tests/test_sam3.py), [test_sam3_workflows.py](tests/test_sam3_workflows.py), [test_sam3_comparison.py](tests/test_sam3_comparison.py).

#### 37. `sam3_video`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** VIDEO + G0–G5.
- **First release target:** Same pinned facebook/sam3 bundle, composite offline/streaming video workflow.
- **Model-specific work:**
  - Replay the separate 32-frame CPU-reference fixtures using stable object identity matching and frame accounting.
  - Test session finalize/reset, cancellation, model-sharing isolation and memory baseline for the advertised lifecycle.
  - Natural multi-object/occlusion/new-arrival claims require new fixtures; retain the failed MPS comparison as failed.
- **Allowed claim after passing review:** At most the documented single-object controlled offline/streaming scope until broader tests pass.
- **Next action:** Replay composite fixtures and identify open lifecycle gates; do not market general tracking quality from the rolled-image clip.
- **Existing implementation/test entry points:** [test_sam3.py](tests/test_sam3.py), [test_sam3_workflows.py](tests/test_sam3_workflows.py), [test_sam3_comparison.py](tests/test_sam3_comparison.py).

### Classification

#### 38. `bert_token_classification`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** NER + G0–G5.
- **First release target:** Pinned dslim/bert-base-NER from native_tasks.json.
- **Model-specific work:**
  - Test token/character alignment, padding masks, label mapping and NONE/SIMPLE aggregation.
  - Replay named entities, no-entity inputs, B1/B2 and the 512-token bound through advertised interfaces.
  - Report strict entity-span precision/recall/F1 using the preserved dataset reconstruction; semantic score quality and execution correctness remain distinct.
- **Allowed claim after passing review:** FP32 NONE/SIMPLE NER within the existing batch/token limits; broader aggregation and quality remain unqualified.
- **Next action:** Restore exact weights, replay span/aggregation/wheel evidence and preserve the recorded modest F1 rather than claiming general high accuracy.
- **Existing implementation/test entry points:** [test_ner_service.py](tests/native_tasks/test_ner_service.py).

#### 39. `clip`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** CLIP + G0–G5.
- **First release target:** Pinned openai/clip-vit-base-patch32, using the audited local safetensors conversion.
- **Model-specific work:**
  - Check image processor, text tokenizer, normalization, candidate prompt template and conversion round-trip integrity.
  - Replay one image with 1–10 labels, top-1 agreement and public interface limits.
  - Retain and reject the failed 100-label profile; do not describe candidate-conditional softmax as calibrated confidence.
- **Allowed claim after passing review:** FP32, one image, 1–10 candidate labels and the fixed documented template.
- **Next action:** Restore/recreate the audited conversion before replay; keep conversion provenance and outstanding weight-license review explicit.
- **Existing implementation/test entry points:** [test_clip_service.py](tests/native_tasks/test_clip_service.py).

#### 40. `distilbert`

- **Starting state:** S from [the dated audit](status_models_testing.md); no promotion issued.
- **Acceptance pack:** CLASSIFY + G0–G5.
- **First release target:** Pinned distilbert/distilbert-base-uncased-finetuned-sst-2-english from native_tasks.json.
- **Model-specific work:**
  - Test masks, label mapping, padded batches and FP32 output against the selected checkpoint.
  - Replay positive/negative sentiment, B1/B2, 512-token bound and Python/CLI/HTTP contracts.
  - Retain the SST-2 794/872 result and source top-1 agreement with exact split/normalization; do not silently replace the evaluator.
- **Allowed claim after passing review:** Bounded FP32 SST-2 sentiment, one or two inputs, maximum 512 tokens.
- **Next action:** Restore deleted model weights to replay, then validate the installed artifact and bounded task rejection behavior.
- **Existing implementation/test entry points:** [test_distilbert_classifier.py](tests/native_tasks/test_distilbert_classifier.py), [test_classification_service.py](tests/native_tasks/test_classification_service.py).

## 8. Proposed release record and evidence bundle

Create one record per **checkpoint + precision + task + interface scope**. This is a proposed record format for planning; the current validators do not automatically accept this schema. Add an explicit schema/validator change during R0 rather than treating this YAML as an existing command contract.

```yaml
schema_version: proposed-mlx-one-model-release-v1
model_type: qwen3
model_id: Qwen/Qwen3-0.6B
revision: "<full immutable checkpoint revision>"
source_commit: "<tested MLX-One commit>"
precision: "<actual weight and execution precision>"
task: text-generation
interfaces: [python, cli, http, sse]
hardware: "<actual chip, unified memory and device>"
software: {python: "<version>", mlx: "<version>", macos: "<version>"}
assets:
  config_sha256: "<hash>"
  processor_sha256: "<hash or explicitly not applicable>"
  weights_manifest_sha256: "<hash>"
conversion: "<identity or recipe/source hashes>"
input_bounds: "<tested token/image/audio/batch limits>"
gates: {G0: PENDING, G1: PENDING, G2: PENDING, G3: PENDING, G4: PENDING, G5: PENDING}
results: []
limitations: []
wheel_sha256: "<tested distribution hash>"
review: {decision: PENDING, reviewer: "<assigned reviewer>"}
```

Store command lines, seeds, real inputs/fixture hashes, generated outputs, actual counts/durations, failures/skips, timing scope and measured memory. A source/reference result needs its own code/weight/processor identity. Publishable evidence must not depend solely on an ignored local directory; private/raw fixtures can be retained separately with reproducible permitted substitutes and clear access limits.

Update existing applicable fields rather than inventing a package-wide support flag:

- `ARCHITECTURE_IMPLEMENTED` stays implementation-only until a real checkpoint passes.
- `MODEL_SMOKE_TESTED` records actual checkpoint execution; it does not promise every public interface.
- For text, `EXPERIMENTAL_SUPPORTED` follows the current functional admission policy. Native tasks retain their existing `decision`, interface, functional, source-parity and release fields; adopt a cross-modality schema only through an explicit reviewed change.
- `USER_VALIDATED` needs independent user evaluation; do not infer it from automated output.
- Failed or unavailable tasks remain BLOCKED/WIP/not implemented as applicable. Do not change all fields merely because another task for the same model passed.
- Existing administrative `alpha_supported`, `release_approval` and publication-environment controls remain separate from functional evidence. This plan grants no release approval.

## 9. Code/package release versus weight release

### Package candidate

1. Freeze the code and release scope; identify admitted profile records and unresolved exclusions.
2. Complete G0–G4 and relevant existing static/backend-free/native checks. Restore only the exact needed weights. A failed or skipped required native test prevents a PASS for that profile.
3. Build wheel and sdist once; validate metadata, payload and notices. Install the wheel in a fresh environment outside the checkout and replay the selected task-specific examples against that installed code.
4. Bind the candidate report to the exact commit and distribution hashes. Review scope wording, examples, failures and changed capabilities.
5. Use the existing candidate-validation and manual publication environment; publish the same tested artifacts after approval. Do not silently rebuild after review.
6. Verify the released package and website display the reviewed scope. Support metadata must not advertise candidates as tested checkpoints.

Upstream MLX-LM publishes on `v*` tags/manual dispatch; the inspected VLM/Audio/Embeddings publishing workflows build distributions on a published release. These packaging jobs are distinct from model tests. MLX-One already has stronger artifact/approval plumbing; following the upstream testing pattern does not require replacing it. See [LM publishing][LM-PUBLISH], [VLM publishing][VLM-PUBLISH], [Audio publishing][AUDIO-PUBLISH], [Embeddings publishing][EMBED-PUBLISH] and [MLX-One's existing workflow](.github/workflows/release.yml).

### Optional Hugging Face weight publication

This is needed only if publishing a converted/quantized checkpoint, not when merely adding library support for an existing publisher repository.

1. Confirm the exact upstream checkpoint and permission to redistribute its weights. Preserve source notices and any gate/access conditions.
2. Save config, weights/shards/index, tokenizer, processor and required special/chat assets as a coherent bundle. Run idempotent sanitization and an independent save/reload task smoke.
3. Draft a model card stating base model/revision, conversion or quantization recipe, actual task, languages where tested, hardware/software, intended use, limitations, license and reproducible native usage.
4. Include evaluation data identities and protocols with any metric; distinguish author-run diagnostics from independent validation. Avoid unsupported leaderboard, listening quality or calibration claims.
5. After explicit publication authorization, upload to the intended repository and record its returned immutable revision. Test loading that published snapshot in a clean environment.
6. Link the published weight revision and library package version; uploading a folder alone does not promote the support status.

Hugging Face references: [model cards][HF-CARD], [uploading models][HF-UPLOAD], [immutable revision downloads][HF-DOWNLOAD], [evaluation overview][HF-EVAL].

## 10. Suggested execution commands and current limits

These examples were **not run** while writing this plan. Use the supported Python/MLX environment for the selected profile, prepare its exact assets, and read the corresponding test's hardware/fixture requirements first.

```bash
# Existing lightweight website metadata checks; these do not test model inference.
python -m unittest discover -s website/tests -v

# Example native contract selections on an Apple Silicon/Metal host.
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest --no-cov \
  tests/models/test_new_language_contracts.py \
  tests/models/test_new_language_native.py

# Examples of modality-specific selections; expand with the per-model entry points.
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest --no-cov \
  tests/test_native_qwen_multimodal_architectures.py \
  tests/test_native_qwen_multimodal_mlx.py

MLX_ONE_RUN_MLX_TESTS=1 python -m pytest --no-cov tests/ocr/
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest --no-cov tests/native_tasks/

# Packaging only, after model/profile gates have passed.
python -m build
python -m twine check dist/*
```

The general release workflow requires the complete applicable native suite, not just these illustrative selections. Test collection and metadata passing do not establish real-checkpoint smoke; ensure the expected real-weight tests ran rather than skipped. Current `scripts/release/wheel_smoke.py` is a text runner with a stale count: extend and validate it as described in section 5 before using it as multimodal evidence. No nonexistent generic `qualify-all-models` command is assumed by this plan.

Resolve missing candidate revisions before fetching weights. `snapshot_download(..., revision=<full SHA>)` is an HF mechanism, not a command to run against every label in a local manifest: **Silero's selected weight identity also relies on its GitHub source commit and asset hash**, and some locally converted models require a conversion recipe rather than an HF download under the internal model label.

## 11. Existing pinned targets to reuse

The following rows are copied from current versioned catalogs as identity pointers, **not approvals**. They have a different checkpoint/task denominator from the 40 registered types. For unpinned candidates above, resolve a full revision and verify actual native compatibility first; do not invent a commit or silently replace the model.

| Catalog / task | Checkpoint or source identity | Recorded revision | Identity source |
| --- | --- | --- | --- |
| Text M01  | `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M02  | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M03  | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M04  | `LiquidAI/LFM2-350M` | `0b6ea934a94b880d1048cbcba45a0d81ca721eb0` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M05  | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M06  | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M07  | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M08  | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M09  | `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M10  | `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M11  | `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | `667408d494bde191c2cb9dd4f4549827816b9275` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M12  | `mlx-community/gpt-oss-20b-MXFP4-Q4` | `f356f2747216d7e98fee755df25987459fc19089` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| Text M13  | `manjunathshiva/gpt-oss-20b-tq3` | `20ba1835e625264e07366d215933fe31a9940a6a` | [Catalog](src/mlx_one/data/compatibility/alpha_release.json) |
| VLM V01  | `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `01af461cdb9574acc09084a0ef94e216e142b085` | [Catalog](docs/benchmarks/vlm/benchmark_plan.json) |
| VLM V02  | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | `46d4cf06a06ffc1a766c214174f9cbed2f45bcab` | [Catalog](docs/benchmarks/vlm/benchmark_plan.json) |
| VLM V03  | `mlx-community/Qwen3.5-0.8B-4bit` | `da28692b5f139cb0ec58a356b437486b7dac7462` | [Catalog](docs/benchmarks/vlm/benchmark_plan.json) |
| VLM V04  | `mlx-community/LFM2-VL-450M-8bit` | `2422c0686f1f1c6267640bc1f069ff4e8b06b8d0` | [Catalog](docs/benchmarks/vlm/benchmark_plan.json) |
| VLM V05  | `LiquidAI/LFM2.5-VL-3B-MLX-4bit` | `7fad79d26092c1145aa8dde873896a07d3e96a3b` | [Catalog](docs/benchmarks/vlm/benchmark_plan.json) |
| Audio A01 transcribe | `openai/whisper-tiny` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A02 transcribe | `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A03 transcribe | `distil-whisper/distil-large-v3` | `8031d2e6ce6631b7fc45629dddfc00271116d981` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A04 transcribe | `UsefulSensors/moonshine-tiny` | `390624ed33d594443aa4aa221f5b9f283b545b5a` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A05 transcribe | `UsefulSensors/moonshine-base` | `7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A06 transcribe | `mlx-community/Qwen3-ASR-0.6B-8bit` | `89e96d92ba34aca20b3e29fb10cc284097d1219f` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A07 transcribe | `mlx-community/Qwen3-ASR-1.7B-8bit` | `a8379a2e2f9e313c9292cdf1af4055ab56d50d55` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A08 transcribe | `nvidia/parakeet-tdt-0.6b-v2` | `ae9ad07059c7c739ffaf932226a8fe64ae2620b0` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A09 transcribe | `mlx-community/parakeet-tdt-0.6b-v3` | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A10 transcribe | `mlx-community/LFM2.5-Audio-1.5B-4bit` | `55d4de65fa10fa935561aaaa995271d8b8ff2ee0` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A11 align | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | `0e1a68e91d815300c7c9754b2a7639378b23db15` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A12 synthesize | `mlx-community/LFM2.5-Audio-1.5B-4bit` | `55d4de65fa10fa935561aaaa995271d8b8ff2ee0` | [Catalog](benchmarks/audio/checkpoints.json) |
| Audio A13 speech-to-speech | `mlx-community/LFM2.5-Audio-1.5B-4bit` | `55d4de65fa10fa935561aaaa995271d8b8ff2ee0` | [Catalog](benchmarks/audio/checkpoints.json) |
| OCR O01 ocr.recognize | `microsoft/trocr-base-printed` | `93450be3f1ed40a930690d951ef3932687cc1892` | [Catalog](docs/benchmarks/ocr/inventory.json) |
| OCR O02 ocr.handwriting | `microsoft/trocr-base-handwritten` | `eaacaf452b06415df8f10bb6fad3a4c11e609406` | [Catalog](docs/benchmarks/ocr/inventory.json) |
| OCR O03 ocr.document | `mlx-community/GOT-OCR2_0-4bit` | `0e28c87720de5d8af67389a0444cf8397524cb61` | [Catalog](docs/benchmarks/ocr/inventory.json) |
| OCR O04 ocr.recognize | `microsoft/Florence-2-base-ft` | `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e` | [Catalog](docs/benchmarks/ocr/inventory.json) |
| OCR O05 ocr.document | `mlx-community/PaddleOCR-VL-1.5-4bit` | `3404726b40ac30b6e4e3d37c504e3f96c9c6c984` | [Catalog](docs/benchmarks/ocr/inventory.json) |
| OCR O06 ocr.document | `lightonai/LightOnOCR-2-1B` | `be7c41313c30e940fe5571de59e5fabc779e00c8` | [Catalog](docs/benchmarks/ocr/inventory.json) |
| OCR O07 ocr.document | `mlx-community/DeepSeek-OCR-4bit` | `4958b6b0d46d7dc3cf27556989dab659996ea426` | [Catalog](docs/benchmarks/ocr/inventory.json) |
| text-classification | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `714eb0fa89d2f80546fda750413ed43d93601a13` | [Native tasks](src/mlx_one/data/native_tasks.json) / [asset lock](docs/native_tasks/checkpoints.lock.yaml) |
| token-classification | `dslim/bert-base-NER` | `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` | [Native tasks](src/mlx_one/data/native_tasks.json) / [asset lock](docs/native_tasks/checkpoints.lock.yaml) |
| zero-shot-image-classification | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | [Native tasks](src/mlx_one/data/native_tasks.json) / [asset lock](docs/native_tasks/checkpoints.lock.yaml) |
| Silero source / selected asset | `snakers4/silero-vad-16k` | `60b7ffa243625ebdc1070275a29f18c87843786a` | [Native tasks](src/mlx_one/data/native_tasks.json) / [asset lock](docs/native_tasks/checkpoints.lock.yaml) |
| SAM3 source checkpoint | `facebook/sam3` | `3c879f39826c281e95690f02c7821c4de09afae7` | [SAM3 evidence](docs/sam3-qualification-m4-32gb.md) |

## 12. Candidates outside the 40-type release denominator

These profiles must not be silently counted as supported registered types. If you choose to release them later, first add/admit their actual runtime and interface contracts and then update the inventory and evidence.

| Candidate | Current boundary | Future release work |
| --- | --- | --- |
| ViT image classification | Experimental helper; intermediate gate failure; no admitted shared task service | Resolve the affected gate and add the declared image-classification interfaces/fixtures. Shared OCR encoder success is not a separate classifier pass. |
| SigLIP2 fixed-resolution classification | BLOCKED | Use the actual fixed-resolution SigLIP checkpoint contract; resolve failed vision trace scope before task admission. Do not treat LFM2-VL's NaFlex component as this classifier's qualification. |
| DETR | BLOCKED, five of twelve recorded image traces fail | Resolve required decoder/logit boundaries and add native object-detection contract, fixtures and public task tests. |
| RT-DETR | Not implemented | Implement and test the architecture/checkpoint and actual detection task; do not transfer a DETR fixture pass. |
| Depth Anything | BLOCKED diagnostic helper | Resolve affected depth/resize gates and admit the actual relative-depth interface; no metric-depth unit claim. |
| Dedicated Qwen2.5-VL document/video QA | BLOCKED tasks on an already-counted VLM type | Keep task status separate; implement missing video-answer workflow and fix the required processor/vision/task gates before exposing a route. |

Other upstream libraries' music, diffusion, voice-cloning or image-generation features are not inherited by MLX-One merely because those projects implement them.

## 13. Done criteria for this plan

A model/profile release ticket is complete only when its target and scope are fixed, required task gates and the installed artifact actually pass, failures/exclusions are recorded, and the reviewer approves the exact claim. Completing architecture code, a timing chart, a documentation card, or an HF upload is not by itself that decision.

This guide supplies **40 per-type plans**, shared gate definitions, modality packs, exact existing target references, proposed evidence fields and release-engineering follow-ups. It deliberately does not mark pending tasks as tested, change current manifests or publish anything.

## Primary-source references

The external code links below are pinned to the researched revision. Hugging Face documentation links are living documentation checked on 8 November 2026. Observed upstream practices in section 2 are separate from the proposed MLX-One gates and per-model work.

Additional task references: [LM architecture tests][LM-TESTS], [Audio WER unit tests][AUDIO-WER], [embedding pooling tests][EMBED-POOLING], [HF ASR][HF-ASR], [HF NER][HF-NER], [HF CLIP task][HF-CLIP], [HF SAM3 video][HF-SAM3].

[LM-CONTRIB]: https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/CONTRIBUTING.md
[LM-TEMPLATE]: https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/.github/PULL_REQUEST_TEMPLATE/new_model.md
[LM-CI]: https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/.github/workflows/pull_request.yml
[LM-PUBLISH]: https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/.github/workflows/release.yml
[LM-TESTS]: https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/tests/test_models.py
[VLM-CONTRIB]: https://github.com/Neywa-Labs/mlx-vlm/blob/a54f1a0d86fa45c466c00d819ac6ac5ba9d61ac4/CONTRIBUTING.md
[VLM-CASES]: https://github.com/Neywa-Labs/mlx-vlm/blob/a54f1a0d86fa45c466c00d819ac6ac5ba9d61ac4/mlx_vlm/tests/model_cases.json
[VLM-CI]: https://github.com/Neywa-Labs/mlx-vlm/blob/a54f1a0d86fa45c466c00d819ac6ac5ba9d61ac4/.github/workflows/tests.yml
[VLM-PUBLISH]: https://github.com/Neywa-Labs/mlx-vlm/blob/a54f1a0d86fa45c466c00d819ac6ac5ba9d61ac4/.github/workflows/python-publish.yml
[AUDIO-ADD]: https://github.com/Neywa-Labs/mlx-audio/blob/ecd43b9f5af55982f4e1bd810d7b60cbc3dfae23/docs/contributing/adding-a-model.md
[AUDIO-CI]: https://github.com/Neywa-Labs/mlx-audio/blob/ecd43b9f5af55982f4e1bd810d7b60cbc3dfae23/.github/workflows/tests.yml
[AUDIO-PUBLISH]: https://github.com/Neywa-Labs/mlx-audio/blob/ecd43b9f5af55982f4e1bd810d7b60cbc3dfae23/.github/workflows/python-publish.yml
[AUDIO-WER]: https://github.com/Neywa-Labs/mlx-audio/blob/ecd43b9f5af55982f4e1bd810d7b60cbc3dfae23/mlx_audio/stt/tests/eval/test_wer.py
[EMBED-TESTS]: https://github.com/Blaizzy/mlx-embeddings/blob/9b28270be81211f2b8daed0041aec65ea5dc4b28/mlx_embeddings/tests/test_models.py
[EMBED-POOLING]: https://github.com/Blaizzy/mlx-embeddings/blob/9b28270be81211f2b8daed0041aec65ea5dc4b28/mlx_embeddings/tests/test_pooling.py
[EMBED-CI]: https://github.com/Blaizzy/mlx-embeddings/blob/9b28270be81211f2b8daed0041aec65ea5dc4b28/.github/workflows/test.yaml
[EMBED-SMOKE]: https://github.com/Blaizzy/mlx-embeddings/blob/9b28270be81211f2b8daed0041aec65ea5dc4b28/mlx_embeddings/tests/test_smoke.py
[EMBED-PUBLISH]: https://github.com/Blaizzy/mlx-embeddings/blob/9b28270be81211f2b8daed0041aec65ea5dc4b28/.github/workflows/python-publish.yaml
[HF-TESTS]: https://huggingface.co/docs/transformers/main/en/testing
[HF-CARD]: https://huggingface.co/docs/hub/model-cards
[HF-DOWNLOAD]: https://huggingface.co/docs/huggingface_hub/guides/download
[HF-UPLOAD]: https://huggingface.co/docs/hub/models-uploading
[HF-EVAL]: https://huggingface.co/docs/evaluate/index
[HF-ASR]: https://huggingface.co/docs/transformers/tasks/asr
[HF-NER]: https://huggingface.co/docs/transformers/tasks/token_classification
[HF-CLIP]: https://huggingface.co/docs/transformers/tasks/zero_shot_image_classification
[HF-SAM3]: https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video
