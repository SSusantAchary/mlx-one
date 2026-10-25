# Model performance charts

Two views use the same validated evidence:

- [Size-versus-speed scatter plot](text-model-size-vs-decode.svg), with a logarithmic
  parameter axis and all 17 recorded direct-model measurements. Historical source
  BF16 runs are hollow markers; the standardized GPT-2 result is a separate card.
- [Bar chart](text-model-decode-bars.svg), inspired by the supplied visual reference.
  Historical primary configurations are ordered by nominal parameter size, with
  measured decode TPS labeled inside each bar. Colors identify precision.

PNG files with the same names are available for embedding. The README uses the
bar chart. All 13 inventory checkpoints appear, including four with no comparable
direct-model result. Missing values are null, without zero-speed bars or estimates.
Qwen3.5's VLM measurements are not text-only benchmark measurements.

The main comparison is historical **128-input / 16-output**, batch one, one warmup,
three measured repetitions, per-column median, native MLX diagnostic dense with
FP16 KV on Apple M4 / 32 GiB, MLX 0.32.2, macOS 15.7.9. Affine Q4/Q8 use group 64;
BF16/FP32 source precisions remain distinct. Size means nominal total parameter
count, not quantized file size. MoE active counts are labeled in the missing-data
cards. Different architectures and precisions confound any size-only interpretation.

The standardized **128/128** GPT-2 result is shown in a separate card: native MLX
direct dense, last-only logits, FP16 KV, MLX 0.32.2, macOS 15.8.1, one warmup and
three measured repetitions. It is not pooled with the historical comparison.

The separate GPT-OSS **HTTP** card comes from five sequential tool-call requests
after one warmup, with 175 actual prompt and 91 actual generated tokens per request
on Apple M4 / 32 GiB. Its decode rate is the median of the recorded native decode
metrics, not generated tokens divided by total HTTP time. That older report does
not record an immutable local checkpoint revision or MLX version. Both are explicitly
unrecorded; the HTTP result is context, not a pinned direct-model comparison or an
identification of the current inventory revision.

[text-model-performance-data.json](text-model-performance-data.json) contains the
plotted values, exact direct-model checkpoint revisions, actual token counts,
hardware/software/cohort identities, evidence paths and SHA-256 hashes. Full raw
measurements remain in [historical](../historical/), [runs](../runs/), and the
[HTTP report](../../validation/gpt-oss-kilo-2026-10-23.json).

To regenerate from the repository root:

```bash
# Optional plotting dependency; no model downloads or inference.
python -m pip install matplotlib
python -m benchmarks.model_performance.plot
python -m benchmarks.model_performance.report --check
```

The generator validates direct-model evidence before plotting and separately
checks HTTP repetitions, recorded counts, success status and finite decode rates.
It does not derive speeds from requested output budgets or extrapolate missing data.

## VLM and audio charts

The same visual style is used for three additional README charts. Every image is
available as PNG and SVG; links below open the vector versions.

| Chart | Primary measurement | Companion measurements | Separate size view |
| --- | --- | --- | --- |
| [Vision-language](vlm-model-performance.svg) | Text decode tokens/sec | TTFT including image processing; peak MLX memory; actual emitted tokens | [Scatter](vlm-model-size-vs-decode.svg) |
| [Speech recognition](audio-asr-performance.svg) | Input audio seconds processed/sec | Single-clip WER; peak MLX memory; separate alignment latency/WIP card | [Scatter](audio-model-size-vs-asr.svg) |
| [Speech generation](audio-speech-generation.svg) | Generated audio seconds/sec | TTFA; actual output duration; RTF; peak MLX memory | One checkpoint, separate TTS/STS panels |

VLM uses the five pinned primary **VLM-HF-S1** records: one 512 × 512 Hugging Face
image, fixed prompt, native resize/tiling, greedy output up to 128 emitted tokens.
The actual fused input and emitted output counts remain in the chart data and raw
records. Larger images, multi-image and OCR workloads are not pooled into this plot.
TTFT includes preprocessing, vision encoder/projection and language prefill.

ASR uses ten pinned **ASR-S1** records, all with the same English utterance and
reference transcript: 9.9 seconds padded to 10 seconds, mono 16 kHz. WER is a
single-clip diagnostic, not a corpus score or a model accuracy ranking. The separate
Qwen ForcedAligner card reports its actual latency and memory; boundary accuracy
remains unmeasured without independent word timings. Alignment retains WIP status.

TTS-S1 uses the fixed 30-word text and US male voice prompt. STS-HF-Q1 uses a
different 2.04-second spoken question with an unconstrained response. Both generate
24 kHz speech; their actual waveform durations determine throughput. These are
different workloads displayed in separate panels, not a shared model ranking.
The failed primary 10-second STS-S1 case remains visible with no numeric speed.
That failure is not replaced by the successful short-question case. No listening
quality/MOS or size-scaling claim is made from one checkpoint.

All measured multimodal records use native MLX/Metal on Apple M4 / 32 GiB,
MLX 0.32.2, one warmup and five repetitions, batch one, per-column median. VLM and
audio remain distinct execution cohorts. Audio charts use the **direct prepared-PCM
task pipeline**: timing includes frontend/encoder/decoding, and speech generation
includes the loaded waveform codec. Model loading, file decoding and HTTP/Engine
overhead are outside those audio timings. Peak MLX includes resident model/codec
components. Existing historical text, VLM synthetic and audio corpus results remain
separate.

Sizes are **nominal publisher checkpoint/model labels**, not reconstructed total
multimodal parameter counts. The component scope of a 0.6B ASR or 1.5B audio model
can differ from a text-only label. Quantized bytes are never converted into an
assumed parameter count. Published size labels come from the pinned VLM inventory,
the [Whisper card](https://huggingface.co/openai/whisper-tiny),
[Turbo card](https://huggingface.co/openai/whisper-large-v3-turbo),
[Distil-Whisper card](https://huggingface.co/distil-whisper/distil-large-v3),
[Moonshine card](https://huggingface.co/UsefulSensors/moonshine-tiny), and the
explicit Qwen/Parakeet/LFM checkpoint names in the audio inventory. The plotting
data retains each size source. Precision and architecture differences prevent a
size-only causal interpretation of speed.

[multimodal-performance-data.json](multimodal-performance-data.json) retains every
plotted metric, failed/null metrics, actual generated sample counts and rates,
checkpoint revision, workload/cohort identity, and source evidence SHA-256. The
generator uses the existing VLM/audio evidence validators, rejects mixed execution
cohorts, and checks primary checkpoint/revision/precision against the inventory.
It reads the explicit audio publication selection and excludes separate streaming,
Python Engine and full-file runs.

Regenerate all VLM/audio views from the repository root:

```bash
python -m benchmarks.model_performance.plot_multimodal
python -m benchmarks.vlm.readme_generator --check
python -m benchmarks.audio.readme_generator --check
```

Only presentation assets, the plotting script and README/documentation links change;
the recorded measurements and frozen inference/benchmark implementations are retained.
