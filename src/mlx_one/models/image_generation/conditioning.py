"""Pinned Qwen3 prompt conditioning for the FLUX.2 and Z-Image checkpoints."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.language.qwen3.config import Qwen3Config
from mlx_one.models.language.qwen3.reference_math import TransformerBlock
from mlx_one.models.shared.reference_math import attention_mask
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.tokenizers import HFTokenizerAdapter


class ImageConditioner(nn.Module):
    """Qwen3 hidden states with image-specific chat framing and layer selection."""

    def __init__(self, config: dict, tokenizer_root: Path, family: str, max_length: int):
        super().__init__()
        args = Qwen3Config.from_dict(config)
        self.selected = (9, 18, 27) if family == "flux2-klein" else (args.num_hidden_layers - 1,)
        self.flux = family == "flux2-klein"
        self.max_length = max_length
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(max(self.selected))]
        self._tokenizer = HFTokenizerAdapter.from_directory(tokenizer_root)
        self._template = ChatTemplate.from_directory(tokenizer_root, "qwen3", self._tokenizer)
        if self._tokenizer.pad_token_id is None or self._template.template is None:
            raise ValueError("image checkpoint requires the pinned Qwen3 padding and chat assets")

    def tokenize(self, prompt: str) -> tuple[list[int], list[int]]:
        """Produce checkpoint-compatible token IDs and masks for batch size one."""
        rendered = self._template.render(
            [{"role": "user", "content": prompt}], enable_thinking=not self.flux
        )
        ids = self._tokenizer.encode(rendered, add_special_tokens=False, max_length=self.max_length)
        valid = len(ids)
        count = self.max_length - valid
        return ids + [self._tokenizer.pad_token_id] * count, [1] * valid + [0] * count

    def __call__(self, prompt: str) -> mx.array:
        ids, valid = self.tokenize(prompt)
        if self.flux:
            mask = attention_mask(len(ids), padding=mx.array([valid]))
        else:
            # Right-padded positions cannot affect earlier causal outputs in B1.
            ids = ids[: sum(valid)]
            mask = attention_mask(len(ids))
        h = self.embed_tokens(mx.array([ids], dtype=mx.int32))
        selected = []
        for i, layer in enumerate(self.layers, start=1):
            h = layer(h, mask=mask)
            if i in self.selected:
                selected.append(h)
        return mx.concatenate(selected, axis=-1)
