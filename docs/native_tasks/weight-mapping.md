# Weight mapping

T01: the exact config, processor and model files are SHA-256 verified before loading. The existing native ViT tower and classifier require an exact set of source tensor names, shapes and float32 dtypes; the patch convolution weight is transposed OIHW to OHWI. All checkpoint tensors loaded in the local diagnostic. No alternate revisions or quantized layouts are accepted.

T02: exact SHA-256 verification of config, tokenizer config, vocabulary and safetensors, followed by exact source/target tensor name, shape and float32 dtype checks. No tensor renaming or convolution conversion. Both checks run offline and reject missing or altered files. Other tasks: NOT MEASURED.

## M2 checkpoint mappings

NER reuses BERT source parameter names except `.attention.self.` to `.attention.self_attn.`. Source-only pooler weights are validated then omitted, matching HF unexpected keys; all model parameters load strictly in FP32. CLIP isolated conversion round-trips all 400 tensors exactly; two nonpersistent position buffers are validated/omitted, patch conv OIHW→OHWI. Fixed SigLIP loads all 408 FP32 tensors, conv OIHW→OHWI and maps encoder `.mlp.fc*` onto existing VisionBlock `.fc*`; pool-head MLP retains source names. No quantized-source parity is claimed.

## M3 checkpoint mappings

Silero15 tensors: conv OIK→OKI and lstm_cell keys append_l0 for existing PredictorLSTM; no bias fusion. Official pinned JIT learned weights differ; rebind oracle graph to selected published safetensors, verify each mapped tensor. DETR534 stored tensors:4 source-only downsample counters omitted after validation,530 native tensors strict, all floating tensors FP32, conv OIHW→OHWI, original parameter names preserved. Reference loading reports exactly the same4 unused counters and no missing/mismatched keys.

## M5 depth mappings

All287 stored FP32 weights/buffers required, none ignored. Patch/stage/fusion/head convOIHW→OHWI; two reassembly transposed-convsIOHW→OHWI. DINOv2 attention query/key/value/output names map to existing native attention projection names. Layer scales and unused mask token retained and shape-checked. Learned position table remains1370×384 and interpolates only at non-original patch grids. No weight/precision substitutions.
