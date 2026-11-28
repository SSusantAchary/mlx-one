# Batch 1 native model implementations

The T5, ViT, BERT/XLM-R and Nomic encoder equations were adapted to native MLX
from Hugging Face Transformers 4.40.2 and the inspected Nomic publisher sources
at nomic-ai/nomic-bert-2048 revision 7710840340a098cfb869c4f65e87cf2b1b70caca.
Upstream authors include Google, Hugging Face, Facebook/Meta, Nomic AI and
Tri Dao (Nomic publisher model header: Copyright (c) 2022, Tri Dao).
These source implementations are Apache-2.0; see models/UPSTREAM_LICENSE.

BGE-M3 pooling, sparse aggregation and ColBERT scoring follow FlagEmbedding
revision 897baa67c0ffddbfccf4e6776ff55310d67809d4 (MIT). No FlagEmbedding source
module or runtime is packaged. Its independently computed outputs are used only
by development-time qualification tooling.

Kokoro native acoustic and vocoder equations follow hexgrad/kokoro revision
dfb907a02bba8152ca444717ca5d78747ccb4bec, including the StyleTTS2-derived modules
and ISTFTNet components distributed there under Apache-2.0. The English eSpeak
phoneme alphabet mapping in audio/kokoro.py adapts hexgrad/misaki/espeak.py at
fba1236595f2d2bf21d414ba6e57d25256afada3, Apache-2.0. See models/UPSTREAM_LICENSE.
Modifications include native MLX convolutions/recurrent layers/FFT, explicit safe
weight conversion, shared task execution, bounded inputs and local voice assets.

The optional phonemizer-fork and eSpeak NG dependencies retain their own GPL-3.0
terms. They are installed separately through the kokoro extra, not copied into
this distribution. Phoneme-input synthesis does not require these frontend tools.
All neural inference runs in mlx_one, without these tools executing neural models.

Checkpoint and voice terms are distinct from implementation licenses. Selected
checkpoint licenses: BGE-small, BGE-M3 and multilingual-E5-small MIT; the other
seven Batch 1 checkpoints Apache-2.0, as declared in their pinned model cards.
Kokoro af_heart is from the same pinned Apache-2.0 checkpoint repository. No
downloaded checkpoint, publisher voice tensor, or reference framework is bundled.
See docs/models/batch-1-sub-6b/references.md in the source repository for exact
asset revisions, source hashes, adaptations and qualification limitations.
