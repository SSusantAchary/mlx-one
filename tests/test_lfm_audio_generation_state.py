"""Boundary EOS preserves the next text group; sampling owns its random state."""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("native MLX test", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.audio.lfm2_audio.checkpoint_config import DepthformerConfig
from mlx_one.models.audio.lfm2_audio.native import AudioEmbeddingWithNorm, AudioHead, NativeLFMAudio


def test_audio_eos_at_group_boundary_resumes_the_full_text_group():
    class Embedding:
        calls = 0

        def as_linear(self, hidden):
            self.calls += 1
            logits = [0.0] * 140
            logits[5 if self.calls <= 4 else 7] = 10.0
            return mx.array([[logits]])

        def __call__(self, ids):
            return mx.zeros((1, 1, 4))

    class Backbone:
        embed_tokens = Embedding()

        def __call__(self, *args, **kwargs):
            return mx.zeros((1, 1, 4)), None, None

    fake = SimpleNamespace(
        config=SimpleNamespace(interleaved_n_text=2, interleaved_n_audio=1),
        make_cache=lambda: (),
        lfm=Backbone(),
        audio_frame=lambda *a, **kw: mx.full((1, 8), 2048, dtype=mx.int32),
        audio_embedding=lambda codes: mx.zeros((1, 4)),
    )
    events = list(
        NativeLFMAudio.generate(fake, mx.zeros((1, 1, 4)), mode="interleaved", max_tokens=12)
    )
    assert [kind for kind, _ in events] == [
        "text",
        "text",
        "audio_end",
        "text",
        "text",
        "audio_end",
        "end",
    ]


def test_audio_sampling_key_is_reproducible_and_does_not_consume_global_rng():
    config = DepthformerConfig(layers=1, dim=8, num_heads=2, num_kv_heads=1)
    fake = SimpleNamespace(
        config=SimpleNamespace(codebooks=2, depthformer=config),
        depth_linear=nn.Linear(8, 16),
        audio_head=AudioHead(config, 2),
        depth_embeddings=[AudioEmbeddingWithNorm(6, 8) for _ in range(2)],
    )
    hidden = mx.ones((1, 1, 8))
    mx.random.seed(91)
    expected = mx.random.uniform(shape=(3,))
    mx.eval(expected)
    mx.random.seed(91)
    a = NativeLFMAudio.audio_frame(fake, hidden, temperature=1.0, top_k=4, key=mx.random.key(0))
    b = NativeLFMAudio.audio_frame(fake, hidden, temperature=1.0, top_k=4, key=mx.random.key(0))
    observed = mx.random.uniform(shape=(3,))
    mx.eval(a, b, observed)
    assert mx.array_equal(a, b).item()
    assert mx.array_equal(observed, expected).item()
