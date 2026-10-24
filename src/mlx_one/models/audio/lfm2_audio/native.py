"""Publisher LFM Audio modules with the shared native MLX-One LFM2 backbone.

Audio embedding/head adaptation: MIT, Prince Canuma and mlx-audio contributors.
See UPSTREAM.md and UPSTREAM_LICENSE for the frozen source revision.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_hybrid_caches
from mlx_one.models.language.lfm2.model import Lfm2Model

from .checkpoint_config import DepthformerConfig, LFM2AudioConfig
from .conformer import MLP, ConformerEncoder
from .transformer import Depthformer, causal_mask


class AudioEmbedding(nn.Module):
    """Audio token embeddings for multiple codebooks (input).

    Uses a shared embedding space with codebook offsets, following the reference
    implementation: audio_embedding(next_token + codebook_offsets).sum(0)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_codebooks: int = 8,
        tie: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size  # Per-codebook vocab size (2049)
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.tie = tie

        # Shared embedding space: total_vocab = vocab_size * num_codebooks
        # Each codebook uses offset: codebook_i uses indices [i*vocab_size, (i+1)*vocab_size)
        total_vocab = vocab_size * num_codebooks
        self.embedding = nn.Embedding(total_vocab, dim)

        # Precompute codebook offsets: [0, 2049, 4098, ...]
        self._codebook_offsets = mx.array([i * vocab_size for i in range(num_codebooks)])

        self.embedding_norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, total_vocab, bias=False)

    def __call__(self, codes: mx.array) -> mx.array:
        """
        Embed audio codes using shared embedding with codebook offsets.

        Args:
            codes: (B, num_codebooks) or (num_codebooks,) audio codes

        Returns:
            Summed embeddings (B, dim) or (dim,)
        """
        squeeze_batch = codes.ndim == 1
        if squeeze_batch:
            codes = codes[None, :]

        K = codes.shape[1]

        # Apply codebook offsets: each codebook's tokens are offset to use separate region
        # codes[:, i] + i * vocab_size
        offset_codes = codes + self._codebook_offsets[:K]

        # (B, K) -> (B, K, dim) -> sum over K -> (B, dim)
        embedded = self.embedding(offset_codes).sum(axis=1)

        if squeeze_batch:
            return embedded.squeeze(0)

        return embedded


class AudioEmbeddingWithNorm(nn.Module):
    """Per-codebook embedding + output projection (reference ``SharedEmbedding``).

    ``embedding_norm`` belongs on the *output* path only: the reference embeds
    tokens raw and normalizes the hidden state right before ``to_logits``.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def embed(self, x: mx.array) -> mx.array:
        """Embed tokens (no normalization), matching ``SharedEmbedding.embed``."""
        return self.embedding(x)

    def get_logits(self, x: mx.array) -> mx.array:
        return self.to_logits(self.embedding_norm(x))


class AudioHead(nn.Module):
    def __init__(
        self,
        depthformer_config: DepthformerConfig,
        num_codebooks: int = 8,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.depthformer_dim = depthformer_config.dim

        self.depthformer = Depthformer(
            layers=depthformer_config.layers,
            dim=depthformer_config.dim,
            num_heads=depthformer_config.num_heads,
            num_kv_heads=depthformer_config.num_kv_heads,
            rope_theta=depthformer_config.rope_theta,
            tie=depthformer_config.tie,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Predict audio hidden states for a whole sequence at once.

        The depthformer attends over the *codebook* axis (a sequence of length
        ``num_codebooks``), with the LFM time steps folded into the batch.
        Frame-by-frame decoding drives ``self.depthformer`` directly instead, so
        this path is uncached and always sees all ``num_codebooks`` positions.

        Args:
            x: Projected hidden states from ``depth_linear`` (B, L, num_codebooks * dim)

        Returns:
            Hidden states per codebook (B, L, num_codebooks, dim)
        """
        B, L, _ = x.shape

        # (B, L, 8*1024) -> (B*L, 8, 1024): codebooks are the sequence dimension
        x = x.reshape(B * L, self.num_codebooks, self.depthformer_dim)

        mask = causal_mask(self.num_codebooks, x.dtype)
        x, _ = self.depthformer(x, mask=mask)

        return x.reshape(B, L, self.num_codebooks, self.depthformer_dim)


class NativeLFMAudio(nn.Module):
    def __init__(self, config: LFM2AudioConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = ConformerEncoder(config.encoder)

        self.audio_adapter = MLP(
            in_channels=config.encoder.d_model,
            out_channels=config.lfm.hidden_size,
            hidden_dims=config.adapter_hidden_dims,
            use_layer_norm=config.adapter_use_layer_norm,
            dropout=config.adapter_dropout,
        )

        self.lfm = Lfm2Model(config.lfm)

        self.audio_embedding = AudioEmbedding(
            config.audio_vocab_size,
            config.lfm.hidden_size,
            config.codebooks,
            config.tie_audio_embeddings,
        )

        self.depth_embeddings = [
            AudioEmbeddingWithNorm(config.audio_vocab_size, config.depthformer.dim)
            for _ in range(config.codebooks)
        ]

        self.depth_linear = nn.Linear(
            config.lfm.hidden_size, config.codebooks * config.depthformer.dim
        )

        self.audio_head = AudioHead(config.depthformer, config.codebooks)

    def make_cache(self):
        return make_hybrid_caches(tuple(self.config.lfm.layer_types), self.config.lfm.conv_L_cache)

    def encode(self, features, *, check=None):
        encoded, lengths = self.audio_encoder(features, check=check)
        return self.audio_adapter(encoded), lengths

    def audio_frame(self, hidden, *, temperature=0.0, top_k=0):
        """Autoregressive codebook axis; a fresh depth cache for each time frame."""
        projected = self.depth_linear(hidden).reshape(
            1, 1, self.config.codebooks, self.config.depthformer.dim
        )
        previous = mx.zeros((1, 1, self.config.depthformer.dim), dtype=projected.dtype)
        cache = None
        codes = []
        for i, embedding in enumerate(self.depth_embeddings):
            value, cache = self.audio_head.depthformer(
                projected[:, :, i] + previous, cache=cache, use_cache=True
            )
            logits = embedding.get_logits(value[:, -1]).astype(mx.float32)
            if temperature > 0:
                logits = logits / temperature
                if 0 < top_k < logits.shape[-1]:
                    threshold = mx.sort(logits, axis=-1)[..., -top_k : -top_k + 1 or None]
                    logits = mx.where(logits >= threshold, logits, -float("inf"))
                code = mx.random.categorical(logits)
            else:
                code = mx.argmax(logits, axis=-1)
            codes.append(code)
            previous = embedding.embed(code)[:, None]
        return mx.stack(codes, axis=-1)

    def generate(
        self,
        embeddings,
        *,
        mode="sequential",
        max_tokens=512,
        audio_temperature=0.0,
        audio_top_k=0,
        cancelled=None,
    ):
        """Ordered text tokens/audio frames; caches belong to this generator."""
        if mode not in {"sequential", "interleaved"} or max_tokens < 1:
            raise ValueError("invalid LFM generation mode/token budget")
        cache = self.make_cache()
        current = "text"
        text_done = False
        remaining = self.config.interleaved_n_text
        try:
            hidden, _, _ = self.lfm(None, cache=cache, input_embeddings=embeddings)
            for _ in range(max_tokens):
                if cancelled and cancelled():
                    raise RuntimeError("audio generation cancelled")
                remaining -= 1
                hidden = hidden[:, -1:]
                if current == "text":
                    token = int(mx.argmax(self.lfm.embed_tokens.as_linear(hidden)[0, -1]).item())
                    if token == 7:
                        yield "end", token
                        return
                    yield "text", token
                    next_embedding = self.lfm.embed_tokens(mx.array([[token]]))
                    text_done = text_done or token == 130
                    if (
                        mode == "sequential"
                        and token == 128
                        or (mode == "interleaved" and (remaining <= 0 or text_done))
                    ):
                        current, remaining = "audio", self.config.interleaved_n_audio
                else:
                    frame = self.audio_frame(
                        hidden, temperature=audio_temperature, top_k=audio_top_k
                    )
                    first = int(frame[0, 0].item())
                    if first == 2048:
                        frame = mx.full(frame.shape, 2048, dtype=mx.int32)
                        current = "text"
                    elif bool(mx.any(frame >= 2048).item()):
                        raise ValueError("nonterminal LFM frame contains invalid codec code")
                    elif mode == "interleaved" and remaining <= 0 and not text_done:
                        current, remaining = "text", self.config.interleaved_n_text
                    yield "audio_end" if first == 2048 else "audio", frame
                    next_embedding = self.audio_embedding(frame)[:, None]
                hidden, _, _ = self.lfm(None, cache=cache, input_embeddings=next_embedding)
            raise ValueError("LFM generation did not terminate within its token budget")
        finally:
            for state in cache:
                state.reset()
