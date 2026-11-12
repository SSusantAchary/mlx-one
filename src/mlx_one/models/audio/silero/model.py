"""Native Silero CNN/STFT with the existing MLX-One predictor LSTM.

Source: snakers4/silero-vad 60b7ffa243625ebdc1070275a29f18c87843786a,
tinygrad_model.py and selected published 16-kHz safetensors; MIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_one.models.audio.parakeet.model import PredictorLSTM
from mlx_one.models.audio.silero.config import SileroConfig
from mlx_one.utils.hashing import sha256_file

REVISION = "60b7ffa243625ebdc1070275a29f18c87843786a"
MODEL_ID = "snakers4/silero-vad-16k"
WEIGHT_SHA256 = "c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1"


class SileroModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SileroConfig()
        self.stft_conv = nn.Conv1d(1, 258, 256, stride=128, bias=False)
        self.conv1 = nn.Conv1d(129, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1)
        self.lstm_cell = PredictorLSTM(self.config)
        self.final_conv = nn.Conv1d(128, 1, 1)

    def __call__(self, x, state=None, *, trace=False):
        if x.ndim != 2 or x.shape[1] != 576 or x.dtype != mx.float32:
            raise ValueError("Silero core input must be FP32 [batch,576] (64 context + 512 PCM)")
        # ReflectionPad1d((0,64)): excludes the final sample at the reflected edge.
        padded = mx.concatenate((x, x[:, -65:-1][:, ::-1]), axis=1)
        spectrum = self.stft_conv(padded[:, :, None])
        x = mx.sqrt(spectrum[:, :, :129] ** 2 + spectrum[:, :, 129:] ** 2)
        stages = {"stft": spectrum.transpose(0, 2, 1), "magnitude": x.transpose(0, 2, 1)}
        for i in range(1, 5):
            x = nn.relu(getattr(self, f"conv{i}")(x))
            if trace:
                stages[f"conv{i}"] = x.transpose(0, 2, 1)
        hidden = None if state is None else (state[0][None], state[1][None])
        h, (hs, cs) = self.lstm_cell(x[:, 0, :], hidden)
        new_state = mx.stack((hs[0], cs[0]))
        probabilities = mx.sigmoid(self.final_conv(nn.relu(h[:, None, :])))[:, :, 0].mean(
            axis=1, keepdims=True
        )
        if trace:
            return probabilities, new_state, stages
        return probabilities, new_state


@dataclass(frozen=True)
class LoadedSilero:
    model: SileroModel
    model_path: Path
    architecture: str = "silero_vad"
    model_id: str = MODEL_ID
    revision: str = REVISION
    context_length: int = 18750  # at most 600 s at 32 ms per chunk

    @property
    def quantization(self):
        return {}


def load_silero(path, *, revision=REVISION):
    if revision != REVISION:
        raise ValueError("unsupported Silero source/checkpoint revision")
    root = Path(path)
    weights = root / "silero_vad_16k.safetensors"
    if sha256_file(weights) != WEIGHT_SHA256:
        raise ValueError("Silero safetensors checksum mismatch")
    tensors = mx.load(str(weights))
    mapped = {}
    for name, tensor in tensors.items():
        if name.startswith("lstm_cell."):
            name += "_l0"
        if tensor.ndim == 3:
            tensor = tensor.transpose(0, 2, 1)
        mapped[name] = tensor
    model = SileroModel()
    expected = dict(tree_flatten(model.parameters()))
    if set(expected) != set(mapped):
        raise ValueError("Silero tensor inventory mismatch")
    for name, value in mapped.items():
        if value.shape != expected[name].shape or value.dtype != mx.float32:
            raise ValueError(f"Silero tensor shape/dtype mismatch: {name}")
    model.load_weights(list(mapped.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return LoadedSilero(model, root.resolve())
