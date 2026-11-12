# Model references and provenance

The checkpoint revision, tensor asset hashes, configuration/processor hashes,
source versions and actual results are retained in the
[evidence directory](../../qualification/batch-1-sub-6b/). A checkpoint name by
itself is not an admission identity. Downloaded weights are not redistributed.

## Exact checkpoint revisions

| ID | Publisher repository and immutable revision | Checkpoint license | Native owner |
| --- | --- | --- | --- |
| B01 | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) | Apache-2.0 | `models/embeddings/bert` |
| B02 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | MIT | `models/embeddings/bert` |
| B03 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181) | MIT | `models/embeddings/xlm_roberta` |
| B04 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42) | Apache-2.0 | `models/embeddings/bert` |
| B05 | [google-t5/t5-small](https://huggingface.co/google-t5/t5-small/tree/df1b051c49625cf57a3d0d8d3863ed4d13564fe4) | Apache-2.0 | `models/language/t5`, `seq2seq` |
| B06 | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224/tree/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3) | Apache-2.0 | `models/classification/vit*`, `classification/image.py` |
| B07 | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) | Apache-2.0 | `models/embeddings/xlm_roberta` |
| B08 | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M/tree/f3ff3571791e39611d31c381e3a41a3af07b4987) | Apache-2.0, including the selected `af_heart` asset | `models/audio/kokoro`, `audio/kokoro.py` |
| B09 | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/e9b6763023c676ca8431644204f50c2b100d9aab) | Apache-2.0 | `models/embeddings/nomic_bert` |
| B10 | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) | MIT | `models/embeddings/bert` |

The [metadata inventories](../../qualification/batch-1-sub-6b/metadata/) retain
Hub/LFS identities and model-card/config hashes. Actual downloaded-byte SHA-256
values and cleanup receipts are recorded per attempt, including failed attempts.

## Independent implementation references

- Transformers **4.40.2**, source commit
  [`4fdf58afb72b0754da30037fc800b6044e7d9c99`](https://github.com/huggingface/transformers/tree/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/models):
  BERT, XLM-R, T5, ViT modeling/configuration/tokenization/processing modules.
  Apache-2.0; per-run JSON also hashes the actual installed oracle source file.
- BGE-M3 dense/sparse/ColBERT formulas:
  [FlagEmbedding `897baa67c0ffddbfccf4e6776ff55310d67809d4`](https://github.com/FlagOpen/FlagEmbedding/tree/897baa67c0ffddbfccf4e6776ff55310d67809d4),
  `inference/embedder/encoder_only/m3.py` and
  `finetune/embedder/encoder_only/m3/modeling.py`; MIT, copyright 2022 staoxiao.
  The oracle implements the inspected head formulas on the pinned HF backbone;
  it does not claim to exercise FlagEmbedding's separate service wrapper.
- Nomic custom publisher encoder:
  [`nomic-ai/nomic-bert-2048` at `7710840340a098cfb869c4f65e87cf2b1b70caca`](https://huggingface.co/nomic-ai/nomic-bert-2048/tree/7710840340a098cfb869c4f65e87cf2b1b70caca),
  `configuration_hf_nomic_bert.py`, `modeling_hf_nomic_bert.py`; Apache-2.0
  repository metadata, including the source header's Tri Dao copyright.
  These inspected local files run only inside the isolated oracle environment.
- Kokoro acoustic/vocoder source:
  [`hexgrad/kokoro` at `dfb907a02bba8152ca444717ca5d78747ccb4bec`](https://github.com/hexgrad/kokoro/tree/dfb907a02bba8152ca444717ca5d78747ccb4bec),
  `model.py`, `modules.py`, `istftnet.py`, `custom_stft.py`; Apache-2.0.
- English phoneme alphabet adaptation:
  [`hexgrad/misaki/espeak.py` at `fba1236595f2d2bf21d414ba6e57d25256afada3`](https://github.com/hexgrad/misaki/blob/fba1236595f2d2bf21d414ba6e57d25256afada3/misaki/espeak.py),
  Apache-2.0. Raw English uses separately installed `phonemizer-fork==3.3.2`
  and `espeakng-loader==0.2.4`; their GPL frontend terms remain distinct from
  Kokoro weights and MLX-One source. No neural fallback is used.

The [reference identity receipt](../../qualification/batch-1-sub-6b/reference-identity.json)
contains source SHA-256 values. The shipped
[notice](../../../src/mlx_one/models/BATCH1_NOTICE.md) retains attribution and the
[FlagEmbedding MIT license](../../../src/mlx_one/models/FLAGEMBEDDING_LICENSE).

## Fixtures and limits

Retrieval uses the campaign-authored `batch1-controlled-retrieval-v1`: three
English/French/German queries and five short documents with one known relevant
answer. The reported Recall@1 and reciprocal rank are **controlled-example
checks**, not MTEB/MS-MARCO evaluation or evidence about every language.

T5 uses two explicit translation prompts, teacher-forced decoder IDs and bounded
deterministic greedy generation. Synthetic tiny-layer fixtures use a deterministic
sine-weight recipe and isolated Transformers outputs; they contain no checkpoint
weights and establish implementation contracts only.

ViT uses rows 0 and 1 from the training Parquet file in
[`AIOmarRehan/AnimalsDataset`](https://huggingface.co/datasets/AIOmarRehan/AnimalsDataset/tree/676c9639be89ef4847325dcdbf141051319129f2),
revision `676c9639be89ef4847325dcdbf141051319129f2`, CC0-1.0. Both are cat images.
The two exact files, labels and hashes are in
[fixtures](../../qualification/batch-1-sub-6b/B06/fixtures/). This is a
processor/logit/top-five parity check, not ImageNet accuracy measurement.

Kokoro uses fixed phonemes for controlled source parity and campaign-authored
English for actual synthesis. Neural source randomness is seeded; controlled
vocoder comparison reuses the publisher harmonic excitation. The human listening
answer and native Whisper round trip refer to the exact retained WAV hash.
The additional Whisper checkpoint and its cleanup are separately recorded.

## Safe offline conversion

Only B03 and B08 require publisher Torch formats. Use an isolated conversion
environment, not the inference environment:

```bash
python -m venv .venv-reference
.venv-reference/bin/python -m pip install \
  torch==2.15.0 transformers==4.40.2 safetensors==0.8.0 numpy pillow attrs loguru
```

Download only the pinned checkpoint assets into owned local directories. For
BGE-M3 include `pytorch_model.bin`, `sparse_linear.pt`, `colbert_linear.pt`,
tokenizer/configuration and sentence-transformer pooling metadata. For Kokoro
include `kokoro-v1_0.pth`, `config.json`, and `voices/af_heart.pt`.

```bash
.venv-reference/bin/python scripts/qualification/batch1_convert.py \
  --kind bge-m3 --source ./bge-m3-publisher --output ./bge-m3-native \
  --inventory docs/qualification/batch-1-sub-6b/metadata/B03/summary.json \
  --report ./bge-m3-conversion.json

.venv-reference/bin/python scripts/qualification/batch1_convert.py \
  --kind kokoro --source ./kokoro-publisher --output ./kokoro-native \
  --inventory docs/qualification/batch-1-sub-6b/metadata/B08/summary.json \
  --shape-contract src/mlx_one/models/audio/kokoro/weight_shapes.json \
  --report ./kokoro-conversion.json
```

The converter verifies publisher tensor bytes against pinned LFS hashes **before**
using `torch.load(weights_only=True)`. It rejects unknown structures, refuses
overwrites, emits safetensors, and records shapes/dtypes/tensor hashes. Native
loaders independently validate the complete expected tensor set.

BGE-M3 retains source head FP16 storage; the FP32 oracle casts the heads to its
backbone dtype. Kokoro folds weight normalization, combines recurrent biases,
transposes convolution storage, and explicitly supplies identity affines only for
the publisher's non-affine instance norms. Those constants are identified in the
conversion report. Its 587 native tensors are not a one-to-one source parameter
count. The original conversion receipt's `source_tensor_elements` field counts
converted elements; newer converters call it `converted_tensor_elements`.

## Reproduce qualification

`scripts/qualification/batch1_campaign.py` owns downloads one checkpoint at a
time, verifies pins, launches separate oracle/native processes, records results
and removes each owned checkpoint in `finally` after persisting evidence.
`--installed` removes `PYTHONPATH`, verifies package location and missing reference
distributions, and runs real CLI/actor/HTTP checks. Keep evidence outside `--work`.

```bash
python scripts/qualification/batch1_campaign.py \
  --work /tmp/mlx-one-batch1-owned \
  --metadata docs/qualification/batch-1-sub-6b/metadata \
  --evidence /tmp/mlx-one-batch1-results \
  --native-python /path/to/wheel-env/bin/python \
  --reference-python /path/to/reference-env/bin/python \
  --nomic-reference /path/to/inspected/nomic-bert-source \
  --image-fixtures docs/qualification/batch-1-sub-6b/B06/fixtures \
  --installed --models B01 B02 B03 B04 B05 B06 B07 B09 B10
```

For B08, run `batch1_kokoro_reference.py` against the pinned local source and
publisher assets, then `batch1_kokoro_native.py --check-interfaces` against the
converted native bundle. The native worker records 100/500-character latency,
RTF, stage parity, controlled vocoder diagnostics, English audio and boundaries.
Human listening remains an independent recorded action, never an automatic PASS.

An overall diagnostic FAIL in a retained attempt stays failed. The engineering
report explains any later functional decision and links the exact supporting
measurements. No reference framework is imported by the installed native worker.
