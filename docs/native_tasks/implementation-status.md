# Native task implementation status

| Milestone | Task | Decision | Evidence / next gate |
|---|---|---|---|
| M1 | T02 DistilBERT SST-2 text classification | EXPERIMENTAL | Shared actor, Python/CLI/HTTP, FP32 parity, full validation-set evaluation and clean-wheel checks complete; release review and broader regimes remain open |
| M1 | T01 ViT image classification | EXPERIMENTAL | First failing block/operator classified; intermediate FP32 gate still fails, so task-service admission is blocked |
| M2 | T03 BERT NER | EXPERIMENTAL | FP32 B1/B2 source/entity parity, 100-sentence quality and shared Python/CLI/HTTP; strict F1 60.36%, matching HF |
| M2 | T04 CLIP | EXPERIMENTAL | Audited conversion, 6 real images, 2/10-label whole traces, shared APIs; 100-label profile fails and is rejected |
| M2 | T04 SigLIP2 | BLOCKED | Fixed-resolution native real checkpoint runs; vision layer 11/12 and final norm fail intermediate FP32; task APIs withheld |
| M3 | T05 Silero VAD | EXPERIMENTAL | Full real-audio state/probability and timestamp parity; shared Python/CLI/HTTP; private streaming sessions; clip-presence quality evaluated |
| M3 | T06 DETR | BLOCKED | Native real checkpoint and box diagnostic run; five of12 COCO traces fail fixed FP32, so APIs withheld |
| M3 | T06 RT-DETR | NOT IMPLEMENTED | DETR prerequisite qualification gate failed; no r50/r18 substitution |
| M4 | T07 document QA | BLOCKED | Internal one-page helper reuses existing VLM; processor passes; dense FP16 vision fails;5/6 authored questions correct; no task APIs |
| M4 | T08 video QA | BLOCKED | Native frame/patch candidate and two source-verified temporal math fixes; real video pixel parity and dense vision gate fail; full QA not admitted |
| M5 | T09 depth estimation | BLOCKED | Native287-tensor DINOv2/reassembly/fusion/head runs; all6 whole traces fail FP32,5/6 final maps pass; no true-depth labels or task APIs |

Milestones M1–M5 are reported in phase-reports/. Implementation and interface availability are distinct from release qualification. MLXBoost remains frozen.
