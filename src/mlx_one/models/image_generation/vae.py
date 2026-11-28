"""Native VAE decoder with FLUX.2 and Z-Image latent normalization.

Reference equations: Diffusers AutoencoderKLFlux2 and AutoencoderKL, and the
MFLUX family decoders. See NOTICE for source revisions and licenses.
"""

import mlx.core as mx
import mlx.nn as nn

from .math import attention, split_heads


def _norm(groups, channels):
    return nn.GroupNorm(groups, channels, eps=1e-6, pytorch_compatible=True)


class _Resnet(nn.Module):
    def __init__(self, source, target, groups):
        super().__init__()
        self.norm1 = _norm(groups, source)
        self.conv1 = nn.Conv2d(source, target, 3, padding=1)
        self.norm2 = _norm(groups, target)
        self.conv2 = nn.Conv2d(target, target, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(source, target, 1) if source != target else None

    def __call__(self, x):
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)
        h = self.conv1(nn.silu(self.norm1(x)))
        return residual + self.conv2(nn.silu(self.norm2(h)))


class _Attention(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.group_norm = _norm(groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x):
        h = self.group_norm(x).reshape(x.shape[0], -1, x.shape[-1])
        q, k, v = [split_heads(proj(h), 1) for proj in (self.to_q, self.to_k, self.to_v)]
        return x + self.to_out[0](attention(q, k, v)).reshape(x.shape)


class _Mid(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.resnets = [_Resnet(channels, channels, groups) for _ in range(2)]
        self.attentions = [_Attention(channels, groups)]

    def __call__(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class _Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        return self.conv(mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2))


class _UpBlock(nn.Module):
    def __init__(self, source, target, layers, groups, upsample):
        super().__init__()
        self.resnets = [_Resnet(source, target, groups)] + [
            _Resnet(target, target, groups) for _ in range(layers - 1)
        ]
        self.upsamplers = [_Upsample(target)] if upsample else []

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for upsample in self.upsamplers:
            x = upsample(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = list(reversed(config["block_out_channels"]))
        groups = config["norm_num_groups"]
        self.conv_in = nn.Conv2d(config["latent_channels"], channels[0], 3, padding=1)
        self.mid_block = _Mid(channels[0], groups)
        self.up_blocks = [
            _UpBlock(source, target, config["layers_per_block"] + 1, groups, i < len(channels) - 1)
            for i, (source, target) in enumerate(
                zip([channels[0], *channels[:-1]], channels, strict=True)
            )
        ]
        self.conv_norm_out = _norm(groups, channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], config["out_channels"], 3, padding=1)

    def __call__(self, x):
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class ImageDecoder(nn.Module):
    """Decode family latents into NHWC RGB in the VAE output range."""

    def __init__(self, config: dict):
        super().__init__()
        self.decoder = _Decoder(config)
        channels = config["latent_channels"]
        if config["use_post_quant_conv"]:
            self.post_quant_conv = nn.Conv2d(channels, channels, 1)
            self.bn = {"running_mean": mx.zeros(channels * 4), "running_var": mx.ones(channels * 4)}
            self.eps = config["batch_norm_eps"]
        else:
            self.scale = config["scaling_factor"]
            self.shift = config["shift_factor"]

    def __call__(self, x: mx.array, rows: int, columns: int) -> mx.array:
        dtype = self.decoder.conv_in.weight.dtype
        if hasattr(self, "bn"):
            mean = self.bn["running_mean"].astype(dtype)
            std = mx.sqrt(self.bn["running_var"].astype(mx.float32) + self.eps).astype(dtype)
            x = x.astype(dtype) * std + mean
            x = x.reshape(1, rows, columns, -1, 2, 2).transpose(0, 1, 4, 2, 5, 3)
            x = x.reshape(1, rows * 2, columns * 2, -1)
            x = self.post_quant_conv(x)
        else:
            x = x.astype(dtype) / self.scale + self.shift
        return self.decoder(x)
