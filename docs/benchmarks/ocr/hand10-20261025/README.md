# Native OCR: 10 handwriting lines

This small experiment uses ten original line images from the **test** split of
[toghrultahirov/handwritten_text_ocr](https://huggingface.co/datasets/toghrultahirov/handwritten_text_ocr/tree/3762d848764659233b02278e97751de068a1eebf).
The split has 2,915 examples. Selection is without replacement using
`random.Random(20261025).sample(range(2915), 10)` in original Parquet order.

`sample.json` retains the immutable dataset revision, global test row indices, source Parquet
files/row groups, image hashes/dimensions and ground-truth `text`. Only `image` and `text`
columns are fetched from selected row groups; alternate OCR engine predictions and processed
image arrays are not used. No image is changed or replaced based on inference output.
The pinned card declares no license; original images are temporary and are not redistributed.

All seven OCR checkpoints are tested on the same ten images. TrOCR Handwritten uses
`ocr.handwriting`; TrOCR Printed uses `ocr.recognize` as an explicitly out-of-domain printed
baseline. The page-capable models use their standard native OCR prompt on each cropped line.
This does not expand the advertised task capabilities of any checkpoint.

Native MLX/Metal, direct-model inference, batch 1, greedy decoding: one 16-token warmup on
the first image, then one timed request per image, a **128-token output cap**, and a 180-second
per-image cooperative timeout. Hardware/software identity, actual output tokens, finish reasons,
truncation counts, per-image predictions and MLX/process memory are retained in each JSON.
Median latency, aggregate images/s, maximum peak MLX allocation and micro CER/WER follow
the [page benchmark definitions](../omni100-20261025/README.md). Handwriting references are
plain text, not Markdown. Failed predictions count as empty output. No page results are pooled
with these line results. Ten examples are a diagnostic sample, not a broad handwriting quality
qualification or a comparison with the dataset's third-party OCR baselines.

`O*.json` contain measurements or explicit failures. `coverage.json` lists every checkpoint;
`cleanup.json` records the separate TrOCR/Florence temporary download cleanup. The other
models reuse their temporary page-benchmark weights and are removed in that campaign's
[cleanup manifest](../omni100-20261025/cleanup.json).

The normal page campaign runs handwriting too when its images have been prepared in the
same scratch directory and this sample manifest is present. To run
handwriting alone, prepare a fresh owned directory and pass the handwriting cohort:

```bash
python -m pip install -e '.[ocr,ocr-bench,viz]'
WORK=$(mktemp -d /tmp/mlx-one-ocr-hand10.XXXXXX)
printf 'hand10-20261025\n' > "$WORK/.mlx-one-ocr-benchmark-owned"
python -m benchmarks.ocr.handwriting --work "$WORK"
python -m benchmarks.ocr.campaign --cohort hand10 --work "$WORK" \
  --ids O01 O02 O03 O04 O05 O06 O07
```

This downloads the pinned checkpoints sequentially and removes the owned models and data
after the run. Archive existing evidence before reusing this cohort's fixed output directory.
