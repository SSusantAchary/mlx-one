# Diagnostic performance

On Apple M4 32 GiB, macOS 15.8.1, Python 3.12.11 and MLX 0.32.2, the unqualified FP32 ViT path processed one generated 319×197 RGB image with a 224×224 processor. The full Python request, including preprocessing and postprocessing, had a median of 24.45 ms and p95 of 25.34 ms over five timed requests after one excluded warmup, in one process. Five observations are insufficient for a stable p95. Peak process RSS was 445,054,976 bytes and peak MLX allocation 510,519,768 bytes; these measures overlap and must not be added. Load time and individual timings are in the ignored local `vit-final-20261102/benchmark.json`. This is not a speedup comparison or qualification claim.

DistilBERT and T03–T09 performance: NOT MEASURED. The current benchmark does not break out forward, preprocessing and postprocessing or provide independent-process estimates.

M1 DistilBERT: one process, one excluded warmup, 20 B1 requests for the fixed sentence in `benchmark_sentiment.py`. Median full Python task latency was **14.952 ms**, p95 **17.579 ms**. Tokenizer/preparation median was 10.791 ms and native forward with array creation/device dispatch median 4.828 ms (p95 8.660 ms). Components were timed separately and do not sum to the full request. Peak RSS was 405,700,608 bytes and peak MLX allocation 275,201,198 bytes; these overlap. No cross-runtime speedup claim is made. Exact samples are in `local/native_tasks/runs/m1-20261103/performance-final.json`.

## M2 diagnostic measurements

Apple M4 32 GiB / macOS15.8.1 / MLX0.32.2 / native FP32, TF32 disabled, one process, B1, one warmup,20 timed requests. NER: request median18.903ms/p9526.142ms; preparation10.761ms; forward including array creation/device dispatch8.053ms; peak RSS584,400,896B, peak MLX441,504,278B. CLIP cat vs dog: request median80.188ms/p9589.683ms; preparation58.631ms; forward with dispatch22.365ms; peak RSS826,884,096B, peak MLX672,054,004B. End-to-end reciprocal median rates are52.90 and12.47 requests/s, respectively, not a separate saturation throughput benchmark. Components timed separately; peaks overlap. No thermal stabilization or speedup claim. Raw20-sample series and environment are in m2-20261103/{ner,clip}-performance.json.

## M3 Silero diagnostics

Same Apple M4/32GiB/nativeMLX0.32.2/FP32 host; one process, one warmup,20 B1 requests. Each request contains first16000 real PCM samples from pinned test.wav (one second), not a512-sample streaming call. Median16.350ms/p9522.965ms; adapter preparation median.009625ms (already decoded PCM), forward including array creation/device dispatch median15.892ms/p9519.852ms. Peak RSS97,681,408B, peakMLX1,788,080B. Reciprocal median61.16 audio-seconds/s is descriptive, not saturation throughput; WAVdecode and network are outside this input profile. No DETR performance claim after failed qualification gate.
