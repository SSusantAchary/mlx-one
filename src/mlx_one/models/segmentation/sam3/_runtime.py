"""Small, inference-only tensor dialect for the SAM3 reference port.

All numerical kernels use MLX. There is no PyTorch/Transformers runtime import.
The dialect preserves reference axis/index semantics so the four cooperating
architectures can be audited against their pinned upstream implementations.
Host synchronization is confined to variable-length indexing and bookkeeping.
"""

from __future__ import annotations

import builtins
import functools
import logging as _logging
import math
from collections import OrderedDict as OrderedDict
from collections import defaultdict as defaultdict
from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from collections.abc import Iterator as Iterator
from copy import deepcopy as deepcopy
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
from tqdm import tqdm as tqdm

try:
    from typing import Unpack as Unpack
except ImportError:  # Python 3.10; annotations in the port are deferred.
    Unpack = dict


def raw(value):
    return value.array if isinstance(value, Tensor) else value


def wrap(value):
    return Tensor(value) if isinstance(value, mx.array) else value


def _shape(args):
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)


def _index(value):
    if isinstance(value, np.ndarray):
        return _index(Tensor(value))
    if isinstance(value, Tensor):
        if value.dtype == mx.bool_:
            return tuple(mx.array(x) for x in np.nonzero(np.asarray(value.array)))
        return value.array
    if isinstance(value, tuple):
        result = []
        for part in value:
            if isinstance(part, Tensor) and part.dtype == mx.bool_:
                result.extend(mx.array(x) for x in np.nonzero(np.asarray(part.array)))
            else:
                result.append(_index(part))
        return tuple(result)
    return value


class Tensor:
    __array_priority__ = 1000

    def __init__(self, value, dtype=None, **kwargs):
        self.array = (
            mx.array(raw(value), dtype=dtype)
            if not isinstance(raw(value), mx.array)
            else raw(value)
        )
        if dtype is not None:
            self.array = self.array.astype(dtype)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def device(self):
        return "mlx"

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def T(self):
        return Tensor(self.array.T)

    def __array__(self, dtype=None, copy=None):
        value = np.asarray(self.array)
        return value.astype(dtype) if dtype is not None else value

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return (Tensor(x) for x in self.array)

    def __bool__(self):
        return builtins.bool(self.item())

    def __int__(self):
        return builtins.int(self.item())

    def __float__(self):
        return builtins.float(self.item())

    def __index__(self):
        return builtins.int(self.item())

    def __getitem__(self, index):
        return Tensor(self.array[_index(index)])

    def __setitem__(self, index, value):
        if self.dtype == mx.int64:
            # Metal cannot scatter int64. IDs/temporal metadata stay exact on CPU.
            def host_index(part):
                if isinstance(part, Tensor):
                    return np.asarray(part)
                if isinstance(part, tuple):
                    return tuple(host_index(x) for x in part)
                return part

            host = np.array(self.array, copy=True)
            host[host_index(index)] = np.asarray(value) if isinstance(value, Tensor) else value
            self.array = mx.array(host)
        else:
            self.array[_index(index)] = raw(value)

    def __repr__(self):
        return f"SAM3Tensor({self.array!r})"

    def __add__(self, x):
        return Tensor(self.array + raw(x))

    __radd__ = __add__

    def __sub__(self, x):
        return Tensor(self.array - raw(x))

    def __rsub__(self, x):
        return Tensor(raw(x) - self.array)

    def __mul__(self, x):
        return Tensor(self.array * raw(x))

    __rmul__ = __mul__

    def __truediv__(self, x):
        return Tensor(self.array / raw(x))

    def __rtruediv__(self, x):
        return Tensor(raw(x) / self.array)

    def __floordiv__(self, x):
        return Tensor(self.array // raw(x))

    def __rfloordiv__(self, x):
        return Tensor(raw(x) // self.array)

    def __pow__(self, x):
        return Tensor(self.array ** raw(x))

    def __rpow__(self, x):
        return Tensor(raw(x) ** self.array)

    def __mod__(self, x):
        return Tensor(self.array % raw(x))

    def __neg__(self):
        return Tensor(-self.array)

    def __invert__(self):
        return Tensor(~self.array)

    def __and__(self, x):
        return Tensor(self.array & raw(x))

    __rand__ = __and__

    def __or__(self, x):
        return Tensor(self.array | raw(x))

    __ror__ = __or__

    def __eq__(self, x):
        return Tensor(self.array == raw(x))

    def __ne__(self, x):
        return Tensor(self.array != raw(x))

    def __lt__(self, x):
        return Tensor(self.array < raw(x))

    def __le__(self, x):
        return Tensor(self.array <= raw(x))

    def __gt__(self, x):
        return Tensor(self.array > raw(x))

    def __ge__(self, x):
        return Tensor(self.array >= raw(x))

    def __matmul__(self, x):
        return Tensor(self.array @ raw(x))

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self):
        return self.ndim

    def numel(self):
        return self.array.size

    def item(self):
        return self.array.item()

    def tolist(self):
        return self.array.tolist()

    def numpy(self):
        return np.asarray(self.array)

    def to(self, device=None, dtype=None, **kwargs):
        if isinstance(device, Tensor):
            dtype = device.dtype
        elif isinstance(device, type(mx.float32)):
            dtype = device
        if isinstance(dtype, str):
            dtype = getattr(mx, dtype)
        if isinstance(device, str) and device == "cpu":
            value = self.array.astype(dtype) if dtype is not None else self.array
            return HostTensor(np.asarray(value))
        return Tensor(self.array.astype(dtype)) if dtype is not None else self

    def type_as(self, x):
        return self.to(dtype=x.dtype)

    def float(self):
        return self.to(dtype=mx.float32)

    def long(self):
        return self.to(dtype=mx.int64)

    def int(self):
        return self.to(dtype=mx.int32)

    def bool(self):
        return self.to(dtype=mx.bool_)

    def cpu(self):
        mx.eval(self.array)
        return self

    def detach(self):
        return self

    def contiguous(self):
        return self

    def clone(self):
        return Tensor(mx.array(self.array))

    def view(self, *shape):
        return Tensor(self.array.reshape(_shape(shape)))

    reshape = view

    def permute(self, *axes):
        return Tensor(self.array.transpose(_shape(axes)))

    def transpose(self, a, b):
        return Tensor(mx.swapaxes(self.array, a, b))

    def unsqueeze(self, dim):
        return Tensor(mx.expand_dims(self.array, dim))

    def squeeze(self, dim=None):
        if dim is not None and self.shape[dim] != 1:
            return self
        return Tensor(mx.squeeze(self.array, axis=dim))

    def flatten(self, start_dim=0, end_dim=-1):
        start_dim %= self.ndim
        end_dim %= self.ndim
        return self.reshape(
            *self.shape[:start_dim],
            math.prod(self.shape[start_dim : end_dim + 1]),
            *self.shape[end_dim + 1 :],
        )

    def expand(self, *shape):
        shape = _shape(shape)
        old = (1,) * (len(shape) - self.ndim) + self.shape
        return Tensor(
            mx.broadcast_to(self.array, tuple(a if b == -1 else b for a, b in zip(old, shape)))
        )

    def expand_as(self, x):
        return self.expand(x.shape)

    def repeat(self, *repeats):
        return Tensor(mx.tile(self.array, _shape(repeats)))

    tile = repeat

    def repeat_interleave(self, repeats, dim=None):
        return Tensor(mx.repeat(self.array, repeats, axis=dim))

    def unbind(self, dim=0):
        return tuple(Tensor(mx.take(self.array, i, axis=dim)) for i in range(self.shape[dim]))

    def split(self, size, dim=0):
        edges = (
            list(np.cumsum(size)[:-1])
            if isinstance(size, (list, tuple))
            else list(range(size, self.shape[dim], size))
        )
        return tuple(Tensor(x) for x in mx.split(self.array, edges, axis=dim))

    def chunk(self, chunks, dim=0):
        return self.split(math.ceil(self.shape[dim] / chunks), dim)

    def sum(self, dim=None, keepdim=False, dtype=None):
        return Tensor(
            mx.sum(self.array.astype(dtype) if dtype else self.array, axis=dim, keepdims=keepdim)
        )

    def mean(self, dim=None, keepdim=False):
        return Tensor(mx.mean(self.array, axis=dim, keepdims=keepdim))

    def prod(self, dim=None, keepdim=False):
        return Tensor(mx.prod(self.array, axis=dim, keepdims=keepdim))

    def cumsum(self, dim, dtype=None):
        return Tensor(mx.cumsum(self.array.astype(dtype) if dtype else self.array, axis=dim))

    def any(self, dim=None, keepdim=False):
        return Tensor(mx.any(self.array, axis=dim, keepdims=keepdim))

    def all(self, dim=None, keepdim=False):
        return Tensor(mx.all(self.array, axis=dim, keepdims=keepdim))

    def max(self, dim=None, keepdim=False):
        values = Tensor(mx.max(self.array, axis=dim, keepdims=keepdim))
        if dim is None:
            return values
        return Reduction(values, Tensor(mx.argmax(self.array, axis=dim, keepdims=keepdim)))

    def min(self, dim=None, keepdim=False):
        values = Tensor(mx.min(self.array, axis=dim, keepdims=keepdim))
        if dim is None:
            return values
        return Reduction(values, Tensor(mx.argmin(self.array, axis=dim, keepdims=keepdim)))

    def argmax(self, dim=None):
        return Tensor(mx.argmax(self.array, axis=dim))

    def argmin(self, dim=None):
        return Tensor(mx.argmin(self.array, axis=dim))

    def gather(self, dim, index):
        return ops.gather(self, dim, index)

    def floor(self):
        return Tensor(mx.floor(self.array))

    def topk(self, k, dim=-1, largest=True, sorted=True):
        indices = mx.argsort(-self.array if largest else self.array, axis=dim)
        indices = mx.take(indices, mx.arange(k), axis=dim)
        return Reduction(Tensor(mx.take_along_axis(self.array, indices, axis=dim)), Tensor(indices))

    def nonzero(self, as_tuple=False):
        values = np.nonzero(np.asarray(self.array))
        return (
            tuple(Tensor(mx.array(x)) for x in values)
            if as_tuple
            else Tensor(mx.array(np.stack(values, axis=-1)))
        )

    def masked_fill(self, mask, value):
        return Tensor(mx.where(raw(mask), raw(value), self.array))

    def clamp(self, min=None, max=None):
        a = self.array
        if min is not None:
            a = mx.maximum(a, raw(min))
        if max is not None:
            a = mx.minimum(a, raw(max))
        return Tensor(a)

    def clamp_(self, min=None, max=None):
        self.array = self.clamp(min, max).array
        return self

    def floor_divide_(self, other):
        self.array = self.array // raw(other)
        return self

    def zero_(self):
        self.array = mx.zeros_like(self.array)
        return self

    zeros_ = zero_

    def normal_(self, mean=0, std=1):
        self.array = mx.random.normal(self.shape) * std + mean
        return self

    def fill_(self, value):
        self.array = mx.full(self.shape, value, dtype=self.dtype)
        return self

    def copy_(self, other):
        self.array = mx.array(raw(other))
        return self

    def sigmoid(self):
        return Tensor(mx.sigmoid(self.array))

    def softmax(self, dim=-1, dtype=None):
        return Tensor(mx.softmax(self.array.astype(dtype) if dtype else self.array, axis=dim))

    def sin(self):
        return Tensor(mx.sin(self.array))

    def cos(self):
        return Tensor(mx.cos(self.array))

    def sqrt(self):
        return Tensor(mx.sqrt(self.array))

    def abs(self):
        return Tensor(mx.abs(self.array))

    def new_zeros(self, *shape, **kwargs):
        return ops.zeros(*shape, dtype=kwargs.get("dtype", self.dtype))

    def new_ones(self, *shape, **kwargs):
        return ops.ones(*shape, dtype=kwargs.get("dtype", self.dtype))

    def unique(self, sorted=True):
        return Tensor(mx.array(np.unique(np.asarray(self.array))))


class HostTensor(Tensor):
    """Evaluated CPU-owned history; materialized in MLX only when read."""

    def __init__(self, value):
        self.host = np.array(value, copy=True)

    @property
    def array(self):
        return mx.array(self.host)

    @array.setter
    def array(self, value):
        self.host = np.array(value, copy=True)

    @property
    def shape(self):
        return self.host.shape

    @property
    def dtype(self):
        return getattr(mx, str(self.host.dtype).replace("bool", "bool_"))

    @property
    def device(self):
        return "cpu"

    def __array__(self, dtype=None, copy=None):
        return self.host.astype(dtype) if dtype else self.host

    def to(self, device=None, dtype=None, **kwargs):
        if isinstance(dtype, str):
            dtype = getattr(mx, dtype)
        if device == "cpu" and dtype is None:
            return self
        return Tensor(mx.array(self.host, dtype=dtype))


class Reduction(tuple):
    def __new__(cls, values, indices):
        return super().__new__(cls, (values, indices))

    @property
    def values(self):
        return self[0]

    @property
    def indices(self):
        return self[1]


class Module:
    training = False

    def __init__(self):
        self._nonpersistent_buffers = set()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_buffer(self, name, value, persistent=True):
        setattr(self, name, value)
        if not persistent:
            self._nonpersistent_buffers.add(name)

    def eval(self):
        self.training = False
        for _, m in self.named_modules():
            m.training = False
        return self

    def train(self, mode=True):
        if mode:
            raise RuntimeError("SAM3 currently supports inference only")
        return self.eval()

    def named_modules(self, prefix=""):
        yield prefix, self
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield from value.named_modules(f"{prefix}.{name}".strip("."))
            elif isinstance(value, (list, tuple)):
                for i, child in enumerate(value):
                    if isinstance(child, Module):
                        yield from child.named_modules(f"{prefix}.{name}.{i}".strip("."))

    def state_dict(self):
        result = {}
        for prefix, module in self.named_modules():
            for name, value in vars(module).items():
                if isinstance(value, Tensor) and name not in module._nonpersistent_buffers:
                    result[f"{prefix}.{name}".strip(".")] = value
        return result

    def load_weights(self, weights, strict=True):
        provided = dict(weights)
        expected = self.state_dict()
        if strict and provided.keys() != expected.keys():
            raise ValueError(
                f"weight keys differ: missing={sorted(expected.keys() - provided.keys())}, unexpected={sorted(provided.keys() - expected.keys())}"
            )
        for name, value in provided.items():
            if name not in expected:
                continue
            array = raw(value)
            if tuple(array.shape) != expected[name].shape:
                raise ValueError(f"{name}: expected {expected[name].shape}, got {array.shape}")
            expected[name].array = array.astype(mx.float32)

    @property
    def dtype(self):
        return mx.float32

    @property
    def device(self):
        return "mlx"

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        return {name: value.array for name, value in self.state_dict().items()}


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(mx.zeros((out_features, in_features)))
        self.bias = Tensor(mx.zeros((out_features,))) if bias else None

    def forward(self, x):
        value = raw(x) @ self.weight.array.T
        return Tensor(value + self.bias.array if self.bias is not None else value)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__()
        self.weight = Tensor(mx.zeros((num_embeddings, embedding_dim)))

    def forward(self, x):
        return Tensor(self.weight.array[raw(x)])


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        )
        self.eps = eps
        self.weight = Tensor(mx.ones(self.normalized_shape)) if elementwise_affine else None
        self.bias = Tensor(mx.zeros(self.normalized_shape)) if elementwise_affine else None

    def forward(self, x):
        return Tensor(mx.fast.layer_norm(raw(x), raw(self.weight), raw(self.bias), self.eps))


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = Tensor(mx.ones((num_channels,))) if affine else None
        self.bias = Tensor(mx.zeros((num_channels,))) if affine else None

    def forward(self, x):
        a = raw(x)
        b, c, *rest = a.shape
        g = a.reshape(b, self.num_groups, -1)
        mean = mx.mean(g, axis=-1, keepdims=True)
        var = mx.var(g, axis=-1, keepdims=True)
        a = ((g - mean) * mx.rsqrt(var + self.eps)).reshape(a.shape)
        if self.weight is not None:
            a = a * self.weight.array.reshape(1, c, *([1] * len(rest))) + self.bias.array.reshape(
                1, c, *([1] * len(rest))
            )
        return Tensor(a)


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = Tensor(mx.zeros((out_channels, in_channels // groups, *_pair(kernel_size))))
        self.bias = Tensor(mx.zeros((out_channels,))) if bias else None

    def forward(self, x):
        a = mx.conv2d(
            raw(x).transpose(0, 2, 3, 1),
            self.weight.array.transpose(0, 2, 3, 1),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            a = a + self.bias.array
        return Tensor(a.transpose(0, 3, 1, 2))


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        if groups != 1:
            raise ValueError("grouped transpose convolution unsupported")
        self.weight = Tensor(mx.zeros((in_channels, out_channels, *_pair(kernel_size))))
        self.bias = Tensor(mx.zeros((out_channels,))) if bias else None

    def forward(self, x):
        a = mx.conv_transpose2d(
            raw(x).transpose(0, 2, 3, 1),
            self.weight.array.transpose(1, 2, 3, 0),
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        if self.bias is not None:
            a = a + self.bias.array
        return Tensor(a.transpose(0, 3, 1, 2))


class Identity(Module):
    def forward(self, x):
        return x


class Dropout(Identity):
    def __init__(self, p=0.0, **kwargs):
        super().__init__()


class GELU(Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return Tensor(
            mlx_nn.gelu(raw(x)) if self.approximate == "none" else mlx_nn.gelu_approx(raw(x))
        )


class ReLU(Module):
    def forward(self, x):
        return Tensor(mx.maximum(raw(x), 0))


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = padding

    def forward(self, x):
        if self.padding:
            raise ValueError("padded max pooling is unsupported")
        return Tensor(
            mlx_nn.MaxPool2d(self.kernel, self.stride)(raw(x).transpose(0, 2, 3, 1)).transpose(
                0, 3, 1, 2
            )
        )


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, **kwargs):
    a = raw(input)
    h, w = a.shape[-2:]
    if size is None:
        sh, sw = (
            (scale_factor, scale_factor) if isinstance(scale_factor, (int, float)) else scale_factor
        )
        size = (int(h * sh), int(w * sw))
    if isinstance(size, int):
        size = (size, size)
    oh, ow = tuple(int(x) for x in size)
    if mode == "nearest":
        yi = mx.minimum(mx.floor(mx.arange(oh) * h / oh).astype(mx.int32), h - 1)
        xi = mx.minimum(mx.floor(mx.arange(ow) * w / ow).astype(mx.int32), w - 1)
        return Tensor(mx.take(mx.take(a, yi, axis=-2), xi, axis=-1))
    if mode != "bilinear":
        raise ValueError(f"unsupported interpolation mode {mode!r}")
    if kwargs.get("antialias") and (oh < h or ow < w):
        from mlx_one.segmentation.resizing import bilinear_coefficients

        ix, wx = bilinear_coefficients(w, ow)
        iy, wy = bilinear_coefficients(h, oh)
        horizontal = mx.sum(mx.take(a, mx.array(ix), axis=-1) * mx.array(wx), axis=-1)
        vertical = mx.take(horizontal, mx.array(iy), axis=-2)
        weights = mx.array(wy).reshape(*([1] * (a.ndim - 2)), oh, wy.shape[-1], 1)
        return Tensor(mx.sum(vertical * weights, axis=-2))
    ys = mx.arange(oh, dtype=mx.float32) * (
        (h - 1) / (oh - 1) if align_corners and oh > 1 else h / oh
    )
    xs = mx.arange(ow, dtype=mx.float32) * (
        (w - 1) / (ow - 1) if align_corners and ow > 1 else w / ow
    )
    if not align_corners:
        ys = ys + (h / oh - 1) / 2
        xs = xs + (w / ow - 1) / 2
    ys = mx.clip(ys, 0, h - 1)
    xs = mx.clip(xs, 0, w - 1)
    y0 = mx.floor(ys).astype(mx.int32)
    x0 = mx.floor(xs).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)
    wy = (ys - y0).reshape(*([1] * (a.ndim - 2)), oh, 1)
    wx = (xs - x0).reshape(*([1] * (a.ndim - 1)), ow)
    rows = mx.take(a, y0, axis=-2) * (1 - wy) + mx.take(a, y1, axis=-2) * wy
    return Tensor(mx.take(rows, x0, axis=-1) * (1 - wx) + mx.take(rows, x1, axis=-1) * wx)


def pad(input, pad, mode="constant", value=0):
    if mode != "constant":
        raise ValueError("SAM3 only supports constant padding")
    widths = [(0, 0)] * raw(input).ndim
    for i in range(len(pad) // 2):
        widths[-1 - i] = (pad[2 * i], pad[2 * i + 1])
    return Tensor(mx.pad(raw(input), widths, constant_values=value))


functional = SimpleNamespace(
    interpolate=interpolate,
    pad=pad,
    relu=lambda x: Tensor(mx.maximum(raw(x), 0)),
    sigmoid=lambda x: Tensor(mx.sigmoid(raw(x))),
    dropout=lambda x, **k: x,
    softmax=lambda x, dim=-1, dtype=None: x.softmax(dim, dtype),
)
layers = SimpleNamespace(
    Module=Module,
    ModuleList=list,
    Parameter=Tensor,
    Linear=Linear,
    Embedding=Embedding,
    Conv2d=Conv2d,
    ConvTranspose2d=ConvTranspose2d,
    LayerNorm=LayerNorm,
    GroupNorm=GroupNorm,
    MaxPool2d=MaxPool2d,
    GELU=GELU,
    ReLU=ReLU,
    Dropout=Dropout,
    Identity=Identity,
    functional=functional,
)


class _NoGrad:
    def __call__(self, fn):
        return fn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Ops:
    Tensor = FloatTensor = BoolTensor = LongTensor = Tensor
    float32 = mx.float32
    float16 = mx.float16
    bfloat16 = mx.bfloat16
    bool = mx.bool_
    int = mx.int32
    int32 = mx.int32
    int64 = long = mx.int64
    uint8 = mx.uint8
    dtype = type(mx.float32)
    device = str
    Size = tuple
    nn = layers
    jit = SimpleNamespace(script=lambda fn: fn, is_tracing=lambda: False)

    def tensor(self, data, dtype=None, **kwargs):
        return Tensor(mx.array(raw(data), dtype=dtype))

    def zeros(self, *shape, dtype=mx.float32, **kwargs):
        return Tensor(mx.zeros(_shape(shape), dtype=dtype))

    empty = zeros

    def ones(self, *shape, dtype=mx.float32, **kwargs):
        return Tensor(mx.ones(_shape(shape), dtype=dtype))

    def full(self, size, fill_value, dtype=mx.float32, **kwargs):
        return Tensor(mx.full(size, fill_value, dtype=dtype))

    def zeros_like(self, x, dtype=None, **kwargs):
        return Tensor(mx.zeros_like(raw(x)).astype(dtype or x.dtype))

    def ones_like(self, x, dtype=None, **kwargs):
        return Tensor(mx.ones_like(raw(x)).astype(dtype or x.dtype))

    def full_like(self, x, fill_value, dtype=None, **kwargs):
        return Tensor(mx.full(x.shape, fill_value, dtype=dtype or x.dtype))

    def randn(self, *shape, **kwargs):
        return Tensor(mx.random.normal(_shape(shape)).astype(kwargs.get("dtype", mx.float32)))

    def arange(self, *args, **kwargs):
        return Tensor(
            mx.arange(
                *(int(x) if isinstance(x, Tensor) else x for x in args), dtype=kwargs.get("dtype")
            )
        )

    def cat(self, tensors, dim=0):
        return Tensor(mx.concatenate([raw(x) for x in tensors], axis=dim))

    def stack(self, tensors, dim=0):
        return Tensor(mx.stack([raw(x) for x in tensors], axis=dim))

    def einsum(self, equation, *tensors):
        return Tensor(mx.einsum(equation, *map(raw, tensors)))

    def matmul(self, a, b):
        return Tensor(mx.matmul(raw(a), raw(b)))

    def outer(self, a, b):
        return Tensor(mx.outer(raw(a), raw(b)))

    def gather(self, x, dim, index):
        return Tensor(mx.take_along_axis(raw(x), raw(index), axis=dim))

    def where(self, condition, x=None, y=None):
        return (
            condition.nonzero(as_tuple=True)
            if x is None
            else Tensor(mx.where(raw(condition), raw(x), raw(y)))
        )

    def nonzero(self, x, as_tuple=False):
        return x.nonzero(as_tuple)

    def isin(self, a, b):
        return Tensor(mx.array(np.isin(np.asarray(raw(a)), np.asarray(raw(b)))))

    def clamp(self, x, min=None, max=None):
        return x.clamp(min, max)

    def max(self, x, other=None, dim=None, keepdim=False):
        return (
            Tensor(mx.maximum(raw(x), raw(other)))
            if isinstance(other, Tensor)
            else x.max(dim if dim is not None else other, keepdim)
        )

    def min(self, x, other=None, dim=None, keepdim=False):
        return (
            Tensor(mx.minimum(raw(x), raw(other)))
            if isinstance(other, Tensor)
            else x.min(dim if dim is not None else other, keepdim)
        )

    def sum(self, x, dim=None, keepdim=False, dtype=None):
        return x.sum(dim, keepdim, dtype)

    def any(self, x, dim=None):
        return x.any(dim)

    def argmax(self, x, dim=None):
        return x.argmax(dim)

    def topk(self, x, k, **kwargs):
        return x.topk(k, **kwargs)

    def triu(self, x, diagonal=0):
        return Tensor(mx.triu(raw(x), k=diagonal))

    def div(self, x, y, rounding_mode=None):
        return x // y if rounding_mode == "floor" else x / y

    def no_grad(self):
        return _NoGrad()

    inference_mode = no_grad

    def __getattr__(self, name):
        if name in ("abs", "sin", "cos", "log", "log2", "sigmoid", "sign", "gt", "lt"):
            fn = getattr(mx, {"gt": "greater", "lt": "less"}.get(name, name))
            return lambda *args: Tensor(fn(*map(raw, args)))
        raise AttributeError(f"unsupported native SAM3 operation {name}")


ops = Ops()


def identity_decorator(obj=None, **kwargs):
    return obj if obj is not None else lambda fn: fn


auto_docstring = capture_outputs = merge_with_config_defaults = identity_decorator


def can_return_tuple(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        return_dict = kwargs.pop("return_dict", True)
        output = fn(*args, **kwargs)
        return output if return_dict else output.to_tuple()

    return call


def compile_compatible_method_lru_cache(maxsize=128):
    # Reference helpers receive scalar tensors for spatial sizes. Avoid caching
    # tensor identities (and retaining their graphs) across unrelated sessions.
    return lambda function: function


TransformersKwargs = FlashAttentionKwargs = dict
OutputRecorder = lambda *args, **kwargs: None
is_flash_attention_requested = lambda config: False
is_kernels_available = lambda: False


class ModelOutput:
    def __getitem__(self, key):
        return getattr(self, key) if isinstance(key, str) else self.to_tuple()[key]

    def to_tuple(self):
        return tuple(
            getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None
        )

    def keys(self):
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    def items(self):
        return [(key, getattr(self, key)) for key in self.keys()]


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: Any = None
    hidden_states: Any = None
    attentions: Any = None


@dataclass
class BaseModelOutputWithPooling(BaseModelOutput):
    pooler_output: Any = None


class PreTrainedModel(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        pass


GradientCheckpointingLayer = Module


def _attention(module, query, key, value, attention_mask=None, scaling=None, **kwargs):
    mask = raw(attention_mask)
    out = mx.fast.scaled_dot_product_attention(
        raw(query), raw(key), raw(value), scale=scaling or query.shape[-1] ** -0.5, mask=mask
    )
    return Tensor(out.transpose(0, 2, 1, 3)), None


class _AttentionRegistry(dict):
    def get_interface(self, name, default):
        return self.get(name, default)


ALL_ATTENTION_FUNCTIONS = _AttentionRegistry(sdpa=_attention)


def create_bidirectional_mask(config, inputs_embeds, attention_mask=None, **kwargs):
    if attention_mask is None:
        return None
    return attention_mask.bool().unsqueeze(1).unsqueeze(1)


def _gelu(x):
    return Tensor(mlx_nn.gelu(raw(x)))


ACT2FN = {
    "gelu": _gelu,
    "relu": functional.relu,
    "silu": lambda x: Tensor(mlx_nn.silu(raw(x))),
    "quick_gelu": lambda x: x * ops.sigmoid(1.702 * x),
}
init = SimpleNamespace(
    normal_=lambda x, mean=0, std=1: x.normal_(mean, std),
    copy_=lambda a, b: a.copy_(b),
    zeros_=lambda x: x.zero_(),
    ones_=lambda x: x.fill_(1),
    constant_=lambda x, v: x.fill_(v),
)


class _Logger:
    def __init__(self, name):
        self.logger = _logging.getLogger(name)
        self.seen = set()

    def __getattr__(self, name):
        if name.endswith("_once"):
            method = getattr(self.logger, name[:-5])

            def once(message, *args):
                if message not in self.seen:
                    self.seen.add(message)
                    method(message, *args)

            return once
        return getattr(self.logger, name)


logging = SimpleNamespace(get_logger=_Logger)


class AutoModel:
    @staticmethod
    def from_config(config, **kwargs):
        from . import model, tracker, tracker_video

        classes = {
            "sam3": model.Sam3Model,
            "sam3_vision_model": model.Sam3VisionModel,
            "sam3_vit_model": model.Sam3ViTModel,
            "sam3_tracker": tracker.Sam3TrackerModel,
            "sam3_tracker_video": tracker_video.Sam3TrackerVideoModel,
        }
        return classes[config.model_type](config, **kwargs)


class _CLIPAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))

    def forward(self, x, mask):
        b, n, c = x.shape
        q, k, v = [
            getattr(self, name)(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2).array
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        a = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(c // self.heads) ** -0.5, mask=raw(mask)
        )
        return self.out_proj(Tensor(a.transpose(0, 2, 1, 3).reshape(b, n, c)))


class _CLIPMLP(Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = Linear(c.intermediate_size, c.hidden_size)
        self.act = ACT2FN[c.hidden_act]

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _CLIPLayer(Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = _CLIPAttention(c)
        self.mlp = _CLIPMLP(c)
        self.layer_norm1 = LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.layer_norm2 = LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, x, mask):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class _CLIPEmbeddings(Module):
    def __init__(self, c):
        super().__init__()
        self.token_embedding = Embedding(c.vocab_size, c.hidden_size)
        self.position_embedding = Embedding(c.max_position_embeddings, c.hidden_size)

    def forward(self, ids):
        return self.token_embedding(ids) + self.position_embedding(ops.arange(ids.shape[-1]))


class _CLIPEncoder(Module):
    def __init__(self, c):
        super().__init__()
        self.layers = [_CLIPLayer(c) for _ in range(c.num_hidden_layers)]

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class _CLIPTextTransformer(Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = _CLIPEmbeddings(c)
        self.encoder = _CLIPEncoder(c)
        self.final_layer_norm = LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, ids, attention_mask):
        n = ids.shape[-1]
        allowed = mx.arange(n)[None, :] <= mx.arange(n)[:, None]
        mask = allowed[None, None, :, :]
        if attention_mask is not None:
            mask = mask & raw(attention_mask).astype(mx.bool_)[:, None, None, :]
        x = self.final_layer_norm(self.encoder(self.embeddings(ids), Tensor(mask)))
        return x


class CLIPTextModelWithProjection(Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.text_model = _CLIPTextTransformer(c)
        self.text_projection = Linear(c.hidden_size, c.projection_dim, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.text_model(input_ids, attention_mask)
        index = (
            input_ids.argmax(-1)
            if self.config.eos_token_id == 2
            else (input_ids == self.config.eos_token_id).int().argmax(-1)
        )
        pooled = hidden[ops.arange(hidden.shape[0]), index]
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden, pooler_output=self.text_projection(pooled)
        )


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    """Native adaptive ROIAlign, preserving torchvision's aligned=False border rules."""
    a = raw(input)
    outputs = []
    ph, pw = _pair(output_size)
    if isinstance(boxes, (list, tuple)):
        parts = []
        for batch, part in enumerate(boxes):
            value = np.asarray(part)
            parts.append(
                np.concatenate([np.full((len(value), 1), batch, dtype=value.dtype), value], axis=-1)
            )
        boxes = ops.tensor(np.concatenate(parts, axis=0))
    for roi in np.asarray(raw(boxes)):
        batch = int(roi[0])
        x0, y0, x1, y1 = roi[1:] * spatial_scale
        offset = 0.5 if aligned else 0.0
        x0 -= offset
        y0 -= offset
        x1 -= offset
        y1 -= offset
        rh = y1 - y0
        rw = x1 - x0
        if not aligned:
            rh = max(rh, 1.0)
            rw = max(rw, 1.0)
        gh = sampling_ratio if sampling_ratio > 0 else max(1, math.ceil(rh / ph))
        gw = sampling_ratio if sampling_ratio > 0 else max(1, math.ceil(rw / pw))
        ys = y0 + (mx.arange(ph)[:, None] + (mx.arange(gh)[None, :] + 0.5) / gh) * (rh / ph)
        xs = x0 + (mx.arange(pw)[:, None] + (mx.arange(gw)[None, :] + 0.5) / gw) * (rw / pw)
        yy = mx.broadcast_to(ys[:, None, :, None], (ph, pw, gh, gw))
        xx = mx.broadcast_to(xs[None, :, None, :], (ph, pw, gh, gw))
        valid = (yy >= -1) & (yy <= a.shape[2]) & (xx >= -1) & (xx <= a.shape[3])
        yy = mx.clip(yy, 0, a.shape[2] - 1)
        xx = mx.clip(xx, 0, a.shape[3] - 1)
        yl = mx.floor(yy).astype(mx.int32)
        xl = mx.floor(xx).astype(mx.int32)
        yh = mx.minimum(yl + 1, a.shape[2] - 1)
        xh = mx.minimum(xl + 1, a.shape[3] - 1)
        ly = yy - yl
        lx = xx - xl
        v = (
            a[batch, :, yl, xl] * (1 - ly) * (1 - lx)
            + a[batch, :, yl, xh] * (1 - ly) * lx
            + a[batch, :, yh, xl] * ly * (1 - lx)
            + a[batch, :, yh, xh] * ly * lx
        )
        outputs.append(mx.mean(v * valid, axis=(-1, -2)))
    return Tensor(mx.stack(outputs)) if outputs else ops.zeros(0, a.shape[1], ph, pw)


torchvision = SimpleNamespace(ops=SimpleNamespace(roi_align=roi_align))


class NativeCV:
    """CPU bookkeeping for mask NMS and eight-connected components, no CUDA kernels."""

    @staticmethod
    def generic_nms(ious, scores, threshold, use_iou_matrix=True):
        matrix = np.asarray(ious)
        order = np.argsort(-np.asarray(scores), kind="stable")
        kept = []
        for i in order:
            if all(matrix[i, j] <= threshold for j in kept):
                kept.append(int(i))
        return ops.tensor(kept, dtype=ops.long)

    @staticmethod
    def cc_2d(mask, get_counts=True):
        data = np.asarray(mask).astype(bool)
        labels = np.zeros(data.shape, dtype=np.int32)
        counts = np.zeros_like(labels)
        for batch in range(data.shape[0]):
            for channel in range(data.shape[1]):
                src = data[batch, channel]
                dst = labels[batch, channel]
                h, w = src.shape
                parents = [0]
                sizes = [0]
                runs = []
                previous = []

                def find(x):
                    while parents[x] != x:
                        parents[x] = parents[parents[x]]
                        x = parents[x]
                    return x

                for y in range(h):
                    transitions = np.diff(np.pad(src[y].astype(np.int8), (1, 1)))
                    starts = np.flatnonzero(transitions == 1)
                    ends = np.flatnonzero(transitions == -1)
                    current = []
                    for start, end in zip(starts, ends):
                        label = len(parents)
                        parents.append(label)
                        sizes.append(int(end - start))
                        for a, b, other in previous:
                            if b >= start and a <= end:
                                root = find(other)
                                own = find(label)
                                if root != own:
                                    parents[root] = own
                                    sizes[own] += sizes[root]
                        current.append((int(start), int(end), label))
                        runs.append((y, int(start), int(end), label))
                    previous = current
                for y, start, end, label in runs:
                    root = find(label)
                    dst[y, start:end] = root
                    counts[batch, channel, y, start:end] = sizes[root]
        return ops.tensor(labels, dtype=ops.int32), ops.tensor(counts, dtype=ops.int32)
