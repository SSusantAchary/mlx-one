# Task quality

M1 evaluated the pinned DistilBERT SST-2 checkpoint on all **872 labeled validation sentences** from `stanfordnlp/sst2`, revision `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`. Native FP32 and HF CPU FP32 eager each classified **794/872 correctly (91.055%)**, with identical top-1 predictions on all 872. The descriptive Wilson 95% interval is 88.976%–92.774%; this fixed validation split is not evidence for arbitrary production distributions. Maximum absolute probability difference was `1.3531e-5`, with no FP32 elementwise failures. Both used paired batches, the same strings, right padding and 512-token truncation.

The evaluation is source-relative and does not claim calibrated confidence, out-of-domain quality or industry-benchmark superiority. No task-quality threshold was retrofitted to promote this model. Its decision remains EXPERIMENTAL. Upstream dataset metadata declares the license unknown; dataset text, oracle outputs and error cases remain ignored local evidence. The package contains no SST-2 sentences. Dataset revision and file hashes are recorded in `checkpoints.lock.yaml`; reference/native per-example scores are in `local/native_tasks/runs/m1-20261103/quality-{hf,native}.json`.

ViT has no real-image quality evaluation because its whole-trace numerical gate remains failed. T03–T09 quality: NOT RUN.

## M2 task quality

NER: first 100 validation sentences from pinned tner/conll2003; source tokens joined by spaces, gold label strings mapped explicitly and evaluated as strict character-span/type matches. 198 gold entities; both native and HF SIMPLE: TP153, FP156, FN45; precision49.51%, recall77.27%, micro entity F1 **60.36%**. This modest result is the actual bounded evaluation, not the checkpoint card metric. Source/native entity decisions match. Distributor concrete reuse terms are unspecified; raw data stays local.

CLIP: six pinned scikit-image real samples, authored coarse subject labels. Two-label accuracy6/6; ten-label accuracy5/6 (rocket mistaken for astronaut by both); 100-label diagnostic5/6 with failed intermediate gate. Top-1 source/native agrees18/18 across all runs. This small set does not establish general zero-shot quality; 100 labels are not admitted. SigLIP2 task quality not run after its numerical gate failed.

## M3 task quality

Silero clip speech-presence diagnostic: first10 cached LibriSpeech test-clean utterances from dataset revision71cacbfb7e2354c4226d01e70d77d5fca3d04ba1, plus authored silence/Gaussian noise/440Hz tone. Both native and same-weight oracle: TP10 FP0 FN0 TN3, clip precision/recall1.0. All segments agree. This is a tiny utterance-presence check; no human segment-boundary annotation, segment F1, music corpus or noisy/multispeaker robustness claim. LibriSpeech authors Panayotov/Chen/Povey/Khudanpur, CC-BY4.0 per https://www.openslr.org/12/; original data and derived PCM remain local with exact hashes.

DETR: COCO image39769,6 real boxes across4 categories. Source/native single-image diagnostic101-point AP50=1, meanAP over IoU0.50:0.95=.9392739; at threshold.9,TP5 FP0 FN1 (IoU>=.5). Same task results despite failed final-logit element. This omits area/crowd stratification and is NOT full COCO mAP. Broader labeled quality and per-photo image license review remain pending; raw images are not redistributed. No RT-DETR quality run.

## M4 bounded document QA quality

Two real pinned DocVQA fixture pages, six question/answer labels authored by visual inspection; not official DocVQA labels. Native candidate5/6 exact match, mean token F1.8333333 under documented word/case normalization. Correct presiding person,room,coffee time,company,year; brand answer was incorrectlyRima instead ofAashirvaad. Full source decoder QA gap was not evaluated after dense vision failed. No video QA quality run; processor/math regressions are not answer-quality evidence. Document reuse license is unspecified; raw pages stay local. Big Buck Bunny excerpt attribution:(c)2008 Blender Foundation/www.bigbuckbunny.org,CC-BY3.0 per https://peach.blender.org/about/; the fixture mirror's Apache2 label is recorded separately.

## M5 source-relative depth metrics and missing true-depth evaluation

Six real images, native versus exact HF checkpoint outputs at original dimensions. Resized-map MAE ranges7.18207e-7–5.32688e-6, RMSE2.22406e-6–8.09406e-6. Five maps pass fixed elementwise gate; grayscale camera map fails12 pixels with maxabs4.71830e-4. All whole model traces fail intermediates, so quality/API promotion stops. These are source-relative reconstruction metrics, not true-depth accuracy. No labeled depth ground truth, calibrated metric-depth errors or broad task quality is claimed.
