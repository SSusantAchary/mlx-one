"""FLUX.2 Klein distilled text-to-image denoiser in checkpoint tensor layout.

Reference equations: Diffusers transformer_flux2.py and MFLUX flux2_transformer.
See NOTICE for source revisions and licenses.
"""

import mlx.core as mx
import mlx.nn as nn

from .math import attention, rotary_angles, split_heads, swiglu, timestep_embedding


class _TimeEmbedding(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.channels = channels
        self.timestep_embedder = {
            "linear_1": nn.Linear(channels, dim, bias=False),
            "linear_2": nn.Linear(dim, dim, bias=False),
        }

    def __call__(self, timestep):
        x = timestep_embedding(timestep, self.channels).astype(timestep.dtype)
        return self.timestep_embedder["linear_2"](nn.silu(self.timestep_embedder["linear_1"](x)))


class _Modulation(nn.Module):
    def __init__(self, dim, count):
        super().__init__()
        self.count = count
        self.linear = nn.Linear(dim, count * dim, bias=False)

    def __call__(self, t):
        return mx.split(self.linear(nn.silu(t))[:, None, :], self.count, axis=-1)


def _condition(x, shift, scale):
    return x * (1 + scale) + shift


class _DualAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.add_q_proj = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.add_v_proj = nn.Linear(dim, dim, bias=False)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_add_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, image, text, angles):
        q, k, v = [
            split_heads(proj(image), self.heads) for proj in (self.to_q, self.to_k, self.to_v)
        ]
        tq, tk, tv = [
            split_heads(proj(text), self.heads)
            for proj in (self.add_q_proj, self.add_k_proj, self.add_v_proj)
        ]
        hidden = attention(
            mx.concatenate((self.norm_added_q(tq), self.norm_q(q)), axis=2),
            mx.concatenate((self.norm_added_k(tk), self.norm_k(k)), axis=2),
            mx.concatenate((tv, v), axis=2),
            angles,
        )
        text_out, image_out = mx.split(hidden, [text.shape[1]], axis=1)
        return self.to_out[0](image_out), self.to_add_out(text_out)


class _FeedForward(nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.linear_in = nn.Linear(dim, 2 * inner, bias=False)
        self.linear_out = nn.Linear(inner, dim, bias=False)

    def __call__(self, x):
        return self.linear_out(swiglu(self.linear_in(x)))


class _DualBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, inner, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = _DualAttention(dim, heads, head_dim, eps)
        self.ff = _FeedForward(dim, inner)
        self.ff_context = _FeedForward(dim, inner)

    def __call__(self, image, text, image_mod, text_mod, angles):
        i_shift, i_scale, i_gate, i_ff_shift, i_ff_scale, i_ff_gate = image_mod
        t_shift, t_scale, t_gate, t_ff_shift, t_ff_scale, t_ff_gate = text_mod
        image_attn, text_attn = self.attn(
            _condition(self.norm(image), i_shift, i_scale),
            _condition(self.norm(text), t_shift, t_scale),
            angles,
        )
        image = image + i_gate * image_attn
        text = text + t_gate * text_attn
        image = image + i_ff_gate * self.ff(_condition(self.norm(image), i_ff_shift, i_ff_scale))
        text = text + t_ff_gate * self.ff_context(
            _condition(self.norm(text), t_ff_shift, t_ff_scale)
        )
        return image, mx.clip(text, -65504, 65504) if text.dtype == mx.float16 else text


class _ParallelAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, inner, eps):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.to_qkv_mlp_proj = nn.Linear(dim, dim * 3 + inner * 2, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(dim + inner, dim, bias=False)

    def __call__(self, x, angles):
        q, k, v, mlp = mx.split(
            self.to_qkv_mlp_proj(x), [self.dim, 2 * self.dim, 3 * self.dim], axis=-1
        )
        q, k, v = [split_heads(value, self.heads) for value in (q, k, v)]
        out = attention(self.norm_q(q), self.norm_k(k), v, angles)
        return self.to_out(mx.concatenate((out, swiglu(mlp)), axis=-1))


class _SingleBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, inner, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = _ParallelAttention(dim, heads, head_dim, inner, eps)

    def __call__(self, x, modulation, angles):
        shift, scale, gate = modulation
        x = x + gate * self.attn(_condition(self.norm(x), shift, scale), angles)
        return mx.clip(x, -65504, 65504) if x.dtype == mx.float16 else x


class _OutputNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x, t):
        scale, shift = mx.split(self.linear(nn.silu(t))[:, None, :], 2, axis=-1)
        return _condition(self.norm(x), shift, scale)


class Flux2Denoiser(nn.Module):
    """Run joint text/image and fused single-stream transformer blocks."""

    def __init__(self, config: dict):
        super().__init__()
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        dim = heads * head_dim
        inner = int(dim * config["mlp_ratio"])
        channels = config["in_channels"]
        eps = config["eps"]
        self.axes = tuple(config["axes_dims_rope"])
        self.theta = config["rope_theta"]
        self.time_guidance_embed = _TimeEmbedding(config["timestep_guidance_channels"], dim)
        self.double_stream_modulation_img = _Modulation(dim, 6)
        self.double_stream_modulation_txt = _Modulation(dim, 6)
        self.single_stream_modulation = _Modulation(dim, 3)
        self.x_embedder = nn.Linear(channels, dim, bias=False)
        self.context_embedder = nn.Linear(config["joint_attention_dim"], dim, bias=False)
        self.transformer_blocks = [
            _DualBlock(dim, heads, head_dim, inner, eps) for _ in range(config["num_layers"])
        ]
        self.single_transformer_blocks = [
            _SingleBlock(dim, heads, head_dim, inner, eps)
            for _ in range(config["num_single_layers"])
        ]
        self.norm_out = _OutputNorm(dim)
        self.proj_out = nn.Linear(dim, config["out_channels"] or channels, bias=False)

    def positions(self, rows: int, columns: int, text_length: int):
        """Build text-first positional axes on the packed latent grid."""
        y, x = mx.meshgrid(mx.arange(rows), mx.arange(columns), indexing="ij")
        image_ids = mx.stack((mx.zeros_like(y), y, x, mx.zeros_like(y)), axis=-1).reshape(-1, 4)
        p = mx.arange(text_length)
        z = mx.zeros_like(p)
        text_ids = mx.stack((z, z, z, p), axis=-1)
        return rotary_angles(mx.concatenate((text_ids, image_ids)), self.axes, self.theta)

    def __call__(self, latents: mx.array, text: mx.array, timestep: float, angles):
        dtype = self.x_embedder.weight.dtype
        if dtype == mx.uint32:
            dtype = self.x_embedder.scales.dtype
        t = self.time_guidance_embed(mx.array([timestep], dtype=dtype))
        image_mod = self.double_stream_modulation_img(t)
        text_mod = self.double_stream_modulation_txt(t)
        single_mod = self.single_stream_modulation(t)
        image, text = (
            self.x_embedder(latents.astype(dtype)),
            self.context_embedder(text.astype(dtype)),
        )
        for block in self.transformer_blocks:
            image, text = block(image, text, image_mod, text_mod, angles)
        hidden = mx.concatenate((text, image), axis=1)
        for block in self.single_transformer_blocks:
            hidden = block(hidden, single_mod, angles)
        return self.proj_out(self.norm_out(hidden[:, text.shape[1] :], t))
