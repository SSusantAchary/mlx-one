"""Z-Image Turbo B1 text-to-image denoiser in checkpoint tensor layout.

Reference equations: Diffusers transformer_z_image.py and MFLUX z_image_transformer.
See NOTICE.md for source revisions and licenses.
"""

import mlx.core as mx
import mlx.nn as nn

from .math import attention, rotary_angles, split_heads, timestep_embedding


class _Attention(nn.Module):
    def __init__(self, dim, heads, eps):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(dim // heads, eps=eps)
        self.norm_k = nn.RMSNorm(dim // heads, eps=eps)

    def __call__(self, x, angles):
        q, k, v = [split_heads(proj(x), self.heads) for proj in (self.to_q, self.to_k, self.to_v)]
        return self.to_out[0](attention(self.norm_q(q), self.norm_k(k), v, angles))


class _FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inner = int(dim / 3 * 8)
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(inner, dim, bias=False)
        self.w3 = nn.Linear(dim, inner, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class _Block(nn.Module):
    def __init__(self, dim, heads, eps, modulated):
        super().__init__()
        self.attention = _Attention(dim, heads, eps)
        self.feed_forward = _FeedForward(dim)
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim)] if modulated else None

    def __call__(self, x, angles, t=None):
        if self.adaLN_modulation is None:
            x = x + self.attention_norm2(self.attention(self.attention_norm1(x), angles))
            return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        sa, ga, sf, gf = mx.split(self.adaLN_modulation[0](t)[:, None, :], 4, axis=-1)
        x = x + mx.tanh(ga) * self.attention_norm2(
            self.attention(self.attention_norm1(x) * (1 + sa), angles)
        )
        return x + mx.tanh(gf) * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * (1 + sf)))


class _TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = [nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, min(dim, 256))]

    def __call__(self, t, dtype):
        x = timestep_embedding(t, 256).astype(dtype)
        return self.mlp[2](self.mlp[1](self.mlp[0](x)))


class _Output(nn.Module):
    def __init__(self, dim, channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.linear = nn.Linear(dim, channels)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(min(dim, 256), dim)]

    def __call__(self, x, t):
        scale = 1 + self.adaLN_modulation[1](self.adaLN_modulation[0](t))[:, None, :]
        return self.linear(self.norm_final(x) * scale)


def _pad_tokens(tokens, pad):
    count = (-tokens.shape[1]) % 32
    return mx.concatenate((tokens, mx.broadcast_to(pad, (1, count, tokens.shape[-1]))), axis=1)


class ZImageDenoiser(nn.Module):
    """Patchify latents and run caption, noise and joint transformation."""

    def __init__(self, config: dict):
        super().__init__()
        self.patch = config["all_patch_size"][0]
        self.channels = config["in_channels"]
        self.key = f"{self.patch}-{config['all_f_patch_size'][0]}"
        patch_dim = self.channels * self.patch**2
        dim, heads, eps = config["dim"], config["n_heads"], config["norm_eps"]
        self.axes = tuple(config["axes_dims"])
        self.theta = config["rope_theta"]
        self.t_scale = config["t_scale"]
        self.all_x_embedder = {self.key: nn.Linear(patch_dim, dim)}
        self.all_final_layer = {self.key: _Output(dim, patch_dim)}
        self.t_embedder = _TimeEmbedding(dim)
        cap_dim = config["cap_feat_dim"]
        self.cap_embedder = [nn.RMSNorm(cap_dim, eps=eps), nn.Linear(cap_dim, dim)]
        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))
        self.noise_refiner = [
            _Block(dim, heads, eps, True) for _ in range(config["n_refiner_layers"])
        ]
        self.context_refiner = [
            _Block(dim, heads, eps, False) for _ in range(config["n_refiner_layers"])
        ]
        self.layers = [_Block(dim, heads, eps, True) for _ in range(config["n_layers"])]

    def positions(self, rows: int, columns: int, text_length: int):
        """Build per-modality rotary axes including learned padding tokens."""
        cap_length = text_length + (-text_length) % 32
        cap_pos = mx.arange(1, cap_length + 1)
        zeros = mx.zeros_like(cap_pos)
        cap_ids = mx.stack((cap_pos, zeros, zeros), axis=-1)
        y, x = mx.meshgrid(mx.arange(rows), mx.arange(columns), indexing="ij")
        image_ids = mx.stack((mx.full(y.shape, cap_length + 1), y, x), axis=-1).reshape(-1, 3)
        image_ids = mx.pad(image_ids, [(0, (-rows * columns) % 32), (0, 0)])
        image_angles = rotary_angles(image_ids, self.axes, self.theta)
        text_angles = rotary_angles(cap_ids, self.axes, self.theta)
        joint = tuple(
            mx.concatenate((a, b)) for a, b in zip(image_angles, text_angles, strict=True)
        )
        return image_angles, text_angles, joint

    def __call__(self, latents: mx.array, text: mx.array, timestep: float, positions):
        image_angles, text_angles, joint = positions
        dtype = self.x_pad_token.dtype
        t = self.t_embedder(mx.array([timestep * self.t_scale], dtype=mx.float32), dtype)
        p = self.patch
        _, height, width, c = latents.shape
        rows, cols = height // p, width // p
        patches = latents.astype(dtype).reshape(1, rows, p, cols, p, c)
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(1, rows * cols, -1)
        image = _pad_tokens(self.all_x_embedder[self.key](patches), self.x_pad_token)
        caption = self.cap_embedder[1](self.cap_embedder[0](text.astype(dtype)))
        caption = _pad_tokens(caption, self.cap_pad_token)
        for layer in self.noise_refiner:
            image = layer(image, image_angles, t)
        for layer in self.context_refiner:
            caption = layer(caption, text_angles)
        hidden = mx.concatenate((image, caption), axis=1)
        for layer in self.layers:
            hidden = layer(hidden, joint, t)
        patches = self.all_final_layer[self.key](hidden[:, : rows * cols], t)
        return (
            patches.reshape(1, rows, cols, p, p, c)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(latents.shape)
        )
