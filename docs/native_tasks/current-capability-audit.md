# Native task expansion: historical starting audit

This is the pre-expansion snapshot. Current outcomes are in implementation-status.md and qualification-matrix.md; statements below describe the initial checkout.

Audited 2026-11-02 at repository HEAD `c0bcfba3b0d19cd50a6b547ab170e89592083438`. The checkout was already dirty, including engine, server, packaging and untracked MLXBoost files. These are user work and are not part of this change. `local/` and `references_libs/` are ignored.

The frozen Transformers commit exists in the sibling repository; source inventory and SHA-256 values are in `sources.lock.yaml`. Silero's v6.2.2 tag resolves to the requested commit. Candidate model revisions are in `checkpoints.lock.yaml`; a revision lock is not proof of a runnable checkpoint. The OpenAI CLIP repository's pinned tree returned no safetensors in the Hub API response; its original pickle checkpoint cannot be loaded directly under the prescribed policy. Silero's pinned tree includes an apparent 16 kHz safetensors representation as well as JIT and ONNX files, but architecture and weights need an audit before use.

| Work | Existing starting point and reuse decision | Initial state |
|---|---|---|
| T01 ViT image classification | `models/ocr/trocr/model.py` contains ViT patch embeddings, blocks and final norm. Reuse only for source-compatible configurations; add classifier, strict loader and source-compatible processor separately. `vision/images.py` has bounded decoding but applies EXIF orientation and alpha compositing, so compare these semantics explicitly. | Encoder exists for OCR; classifier absent |
| T02 DistilBERT classification | No DistilBERT encoder found. `models/shared/encoder.py` offers masks and pooling, but DistilBERT block order and heads require a separate source port. | Absent |
| T03 BERT NER | `models/embeddings/bert` supplies embeddings, bidirectional encoder, masks and checkpoint conventions; verify source equivalence before reusing. Add token head and tokenizer offsets only after parity. | Encoder for embeddings exists; NER absent |
| T04 CLIP/SigLIP2 | `models/segmentation/sam3/_runtime.py` has a CLIP-like text tower, but equivalence to the selected checkpoint is unproven; SigLIP2 has distinct scores and processing. | No qualified classifier |
| T05 Silero | `audio/` and `engine/task_runners/speech.py` provide audio/task infrastructure; no audited Silero state model found. | Absent |
| T06 DETR/RT-DETR | Segmentation and OCR contain vision layers, but no matching detector or box postprocessor was found. | Absent |
| T07 DocQA | `models/vision_language/qwen2_5_vl`, `vision/`, `ocr/` and VLM task runners exist. Must test checkpoint, processor and evidence grounding before advertising DocQA. | VLM/OCR primitives only |
| T08 video text | Qwen2.5-VL model and processor exist; segmentation video handling is a different task and does not establish Qwen video temporal parity. | Not qualified |
| T09 depth | No matching DINOv2/Depth Anything task path found. | Absent |

`engine/contracts.py` defines `TaskKind`; `engine/tasks.py`, `engine/adapter_registry.py` and `engine/task_runners/` implement the shared actor and adapters. `engine/public.py`, `cli.py`, and `server/` are the current public surfaces. They have no validated classifier, detector, VAD, DocQA, video QA or depth endpoints. Add no public server claim before task validation. Model loading patterns include strict safetensors selection and tensor checking in `audio/weights.py` and individual model loaders. Core dependencies include MLX, tokenizers and safetensors; Pillow is a vision extra. Existing numerical gates are `inference_baseline.TOLERANCES` and must stay unchanged. MLXBoost's pre-existing files, flags, defaults and evidence are frozen.
