# Processor and tokenizer parity

T01: one generated RGB non-square 319×197 PNG produced bit-identical float32 input tensors against the slow pinned ViT image processor. MLX-One's bounded decoder also applies EXIF transpose and composites alpha on white; those behaviors differ from an unconditional source RGB conversion and have not been qualified. Other image modes, real images and further processor edge cases remain untested.

T02: 21 fixed cases, including Unicode and one truncated 512-token case, produced identical IDs and masks against the pinned AutoTokenizer for individual requests. M1 adds paired reference traces at max lengths512/32 with Unicode/truncation/right-padding probes; all pass. Offsets are NER-specific and tested separately inM2.

M1 retains all original 21 SST-2 cases and adds 27-case paired-input traces (14 groups, with one final B1), including emoji, HTML-like text, combining Unicode and a long padded/truncated input. IDs, masks, all layers and final outputs pass at max lengths 512 and 32. Request-local tokenizers prevent concurrent max-length settings from mutating the loaded tokenizer. Explicit invalid/whitespace/byte-limit cases are rejected by the public task contract.

## M2 processors

NER uses cased WordPiece with request-local right padding/truncation and exact source special tokens. Structured NONE/SIMPLE entity outputs, original Unicode offsets and source fast-tokenizer decoder grouping match reference goldens in B1/B2. CLIP shortest-edge bicubic resize, floor sizing, center crop, float64 rescale-to-float32 and OpenAI mean/std match exactly on six real images and a generated non-square image; tokenizer IDs/masks match for 2/10/100 labels. Fixed SigLIP2 reuses ViT bilinear 224 resize and mean/std 0.5; pixels and Gemma fixed-64 IDs are exact on the cat fixture. Transparent input is explicitly rejected by the CLIP adapter using an opt-in decoder flag; existing image/OCR behavior remains unchanged.

## M3 processors

Silero admits mono16k PCM/PCM16 WAV only, uses the existing audio input processor with no resampling/mixing, retains512-sample chunks and64-sample lookback, right reflection pad64 for STFT, final zero-pad only at end. Native/rebound oracle full60s PCM,STFT and context match exactly. Timestamp hysteresis/min-speech/min-silence/padding policy adapts pinned pure-Python source with plotting disabled. DETR uses slow pinned bilinear shortest800/longest1333 policy (including unrounded raw-size behavior when long-side limited), ImageNet mean/std, no crop, B1 all-valid pixel mask, nearest feature-mask resize and unclipped absolute xyxy postprocessing. Pixel/mask arrays are exact on cat plus all12 COCO fixtures.

## M4 bounded document/video processors

Document candidate uses the existing native Qwen25ImageProcessor with an explicit max_pixels=28×28×256 task budget and exact checkpoint metadata; existing defaults are untouched. Source/native prompt IDs/grid match exactly and patches pass FP32 on real document0. Single-page image only,64 output-token cap,8192-byte questions, no special-token injection, PDFs/multipage explicitly rejected.

Video candidate reuses existing local FFmpeg frame decoder; shortCFR<=10.1s/25MiB/4MP clips only,2–16 sampled frames, variableFPS/rotation rejected. Real fixture30fps/300frames,8 selected indices[0,37,75,112,150,187,225,262], timestamps=index/30, sampledFPS=.8. Patch packing pairs frames inCTHW, repeats odd final frame, grid exact. Pinned fast video processor resize/normalization differs from reused PIL image processor;11050 pixel failures at unchanged FP32. No video route is admitted.

## M5 checkpoint-selected DPT processing

Actual pinned DPTImageProcessor: bicubic resize, keep_aspect_ratio with scale-as-little-as-possible policy, round to multiple14, no padding, float64 rescale then float32 ImageNet normalization. Native pixels match exactly on all6 real images. Candidate bounds B1/FP32/at most1M processed pixels. DINOv2 position-grid bicubic alignFalse; fusion residual resize bilinear alignFalse, fusion/head resize bilinear alignTrue; output map bicubic alignFalse. Four source operator probes and cat final map test pass; broader model intermediate gates fail. Raw NPZ output is relative depth at original image shape, no metric units/JSON blob API.
