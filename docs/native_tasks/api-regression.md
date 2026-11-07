# Current API and regression evidence

Four experimental task profiles use the existing registry/InferenceService/actor/device owner/model pool/leases and existing CLI/server: DistilBERT text-classification, BERT token-classification, CLIP one-image1–10-label zero-shot classification, and Silero mono16k voice-activity-detection. Python, CLI, authenticated HTTP and clean-wheel checks pass for these specific profiles. Input/error/score/offset/session contracts are in api-contracts.md. No separate inference engine was added; classifiers/VAD reject text generation.

ViT, fixed SigLIP2, DETR, document QA, video QA and depth have no admitted dedicated task routes because recorded numerical/processor/quality gates remain incomplete. Internal document/depth/raw-model helpers are diagnostic, not release support. RT-DETR is not implemented behind its prerequisite gate.

Final M5 default suite: **2238 passed,79 skipped,9 subtests passed**; native task/OCR/VLM/audio/segmentation/MLXBoost selection with both opt-in flags: **157 passed**. Two dependency deprecation warnings per interface run. New implementation/harness lint and git diff --check pass. Earlier milestone results and failures remain sealed separately, including M3 lock-schema correction and sandbox Metal-access reruns.

Clean installed wheel runs with -I outside the checkout, Hub offline and reference packages absent. Parent socket connections denied; actual Python/CLI/HTTP outputs checked, repeated/authenticated calls included. Recursive forbidden inference imports remain empty. Live serve was exercised for M1; later task serve loading/route lifecycle was checked through the installed app. No live-network throughput or broader server release qualification is claimed. Final artifact hashes and precise smoke scope are in the M5 manifest/report.

The wider integration lint audit retains nine pre-existing diagnostics on unchanged lines; proof is in the M5 evidence. This is separate from passing lint on new code/changes.
