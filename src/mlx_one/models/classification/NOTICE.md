Native DistilBERT classification adapts the architecture and inference equations
from Hugging Face Transformers v4.57.1, commit
8cb5963cc22174954e7dca2c0a3320b7dc2f4edc,
src/transformers/models/distilbert/modeling_distilbert.py.

Upstream copyright: 2019-present Hugging Face Inc. team; Apache License 2.0.
See the distribution's Apache-2.0 LICENSE. Modifications: native MLX operations,
strict hash-verified safetensors loading, tokenizers vocabulary loading and shared
MLX-One task-service adapters. Transformers and PyTorch are development-only
references and are not used for production inference.

The selected DistilBERT SST-2 checkpoint is separately published under Apache-2.0.
SST-2 evaluation data has an unspecified license in its distributor metadata;
its raw text is not included in this package.

Native BERT NER, CLIP and fixed-resolution SigLIP adapt the corresponding
Transformers modeling modules at the same commit, and NER/zero-shot task
postprocessing adapts the corresponding pipeline modules. Upstream copyrights:
2018 Google AI Language Team and HuggingFace Inc. (BERT); 2021 The HuggingFace
Team (CLIP); 2023 The HuggingFace Team (SigLIP); pipeline authors HuggingFace
Inc. Apache License 2.0; see models/UPSTREAM_LICENSE. Modifications: MLX-owned
forward and strict checkpoint mapping, reuse of MLX-One encoder primitives,
local tokenizers/image preprocessing, bounded shared task adapters and APIs.

BERT NER checkpoint dslim/bert-base-NER declares MIT in its pinned card.
The fixed google/siglip2-base-patch16-224 checkpoint declares Apache-2.0.
The original OpenAI CLIP repository declares MIT for repository content; the
pinned Hugging Face weight card has no separate license field. Weight-license
review is pending; weights are neither packaged nor redistributed here.
The CLIP offline converter uses verified original weights with Torch
weights_only=True in an isolated reference environment; no converter/runtime
reference framework is imported by production code.

NER evaluation uses local tner/conll2003 data with unspecified concrete reuse
terms (metadata: other). Raw data is not packaged. CLIP evaluation uses six
public-domain/CC0 scikit-image sample images with individual source/hash/license
records in the local evidence manifest and docs/native_tasks/checkpoints.lock.yaml.
