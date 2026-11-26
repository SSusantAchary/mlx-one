# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, mtplx/models/inkling/draft.py.
"""DSpark draft head over committed Inkling target features."""

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import KVCache, _tree_nbytes
from mlx_one.core.sliding_cache import SlidingKVCache

from .model import linear
from .precision import AffineWeight


class DraftState:
    def __init__(self, entries):
        self.entries = entries
        self.feature = None

    def tensors(self):
        return [t for cache in self.entries for t in (cache.keys, cache.values)
                if t is not None] + ([self.feature] if self.feature is not None else [])

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.entries) + _tree_nbytes(self.feature)

    @property
    def allocated_nbytes(self):
        return sum(c.allocated_nbytes for c in self.entries) + _tree_nbytes(self.feature)


class DraftBlock(nn.Module):
    def __init__(self, weights, config, index):
        super().__init__()
        self.w = weights
        self.cfg = config
        self.window = (
            config["sliding_window_size"]
            if index in config["local_layer_ids"]
            else None
        )

    def normalize(self, x, name):
        return mx.fast.rms_norm(x, self.w[name + ".weight"], self.cfg["rms_norm_eps"])

    def heads(self, x, name):
        weight = self.w["self_attn." + name + "_proj.weight"]
        projected = linear(x, weight)
        return projected.reshape(*x.shape[:2], -1, self.cfg["head_dim"]).transpose(
            0, 2, 1, 3
        )

    def rotate(self, x, pos):
        return mx.fast.rope(
            x,
            self.cfg["head_dim"],
            traditional=False,
            base=self.cfg["rope_theta"],
            scale=1.0,
            offset=pos,
        )

    def observe(self, features, entry):
        pos = entry.offset
        keys = self.normalize(self.heads(features, "k_ctx"), "self_attn.k_norm_ctx")
        entry.update(self.rotate(keys, pos), self.heads(features, "v_ctx"))

    def proposal_kv(self, entry, keys, values):
        position = entry.offset
        if (not self.window and position and entry.keys is not None
                and position + keys.shape[2] <= entry.keys.shape[2]
                and entry.keys.dtype == keys.dtype and entry.values.dtype == values.dtype):
            # The append-only bank owns this unused capacity. Leave the logical
            # offset at the committed prefix so the next observation overwrites it.
            try:
                return entry.update(keys, values)
            finally:
                entry._offset = position
        key, value = entry.state()
        return (mx.concatenate([key, keys], axis=2),
                mx.concatenate([value, values], axis=2))

    def attend(self, x, entry):
        position = entry.offset
        q = self.rotate(self.normalize(self.heads(x, "q"), "self_attn.q_norm"), position)
        k = self.rotate(self.normalize(self.heads(x, "k"), "self_attn.k_norm"), position)
        v = self.heads(x, "v")
        if hasattr(entry, "attention"):
            out = entry.attention(q, k, v)
        else:
            key, value = self.proposal_kv(entry, k, v)
            out = mx.fast.scaled_dot_product_attention(
                q, key, value, scale=self.cfg["head_dim"] ** -0.5,
            )
        return linear(
            out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1),
            self.w["self_attn.o_proj.weight"],
        )

    def __call__(self, hidden, residual, entry):
        summed = (
            hidden
            if residual is None
            else hidden.astype(mx.float32) + residual.astype(mx.float32)
        )
        attn = self.attend(
            self.normalize(summed, "input_layernorm").astype(hidden.dtype), entry
        )
        summed = summed.astype(hidden.dtype).astype(mx.float32) + attn.astype(
            mx.float32
        )
        x = self.normalize(summed, "post_attention_layernorm").astype(hidden.dtype)
        gate = linear(x, self.w["mlp.gate_proj.weight"]).astype(mx.float32)
        up = linear(x, self.w["mlp.up_proj.weight"]).astype(mx.float32)
        return linear(
            (nn.silu(gate) * up).astype(x.dtype), self.w["mlp.down_proj.weight"]
        ), summed.astype(hidden.dtype)


class Drafter(nn.Module):
    def __init__(self, weights, config, target):
        super().__init__()
        self.cfg = config
        fuse_weight = weights.pop("fuse_weight")
        self.blend = mx.softmax(fuse_weight.astype(mx.float32), axis=-1).astype(
            fuse_weight.dtype
        )
        self.mask = weights.pop("mask_embed")
        self.output_norm = weights.pop("norm.weight")
        self.blocks = []
        for i in range(config["num_hidden_layers"]):
            stem = f"layers.{i}."
            local = {
                k[len(stem) :]: weights.pop(k)
                for k in tuple(weights)
                if k.startswith(stem)
            }
            # Context projections carry their suffix after 'proj' on disk.
            for name in ("k", "v"):
                local[f"self_attn.{name}_ctx_proj.weight"] = local.pop(
                    f"self_attn.{name}_proj_ctx.weight"
                )
            self.blocks.append(DraftBlock(local, config, i))
        self.markov = {
            k.removeprefix("markov_head."): weights.pop(k)
            for k in tuple(weights)
            if k.startswith("markov_head.")
        }
        if weights:
            raise ValueError(f"Unknown DSpark tensors: {sorted(weights)}")
        self.width = config["block_size"]
        self.vocab = target.cfg["unpadded_vocab_size"]
        self.folded = None

    def make_cache(self, capacity):
        cache = DraftState(tuple(SlidingKVCache(b.window) if b.window else KVCache()
                                 for b in self.blocks))
        if capacity:
            for block, entry in zip(self.blocks, cache.entries, strict=True):
                if not block.window:
                    entry.step = capacity
        return cache

    def mix(self, features, row, weight):
        dtype = features[0].dtype
        mixed = sum(
            h.astype(mx.float32) * self.blend[row, i].astype(mx.float32)
            for i, h in enumerate(features)
        )
        return mx.fast.rms_norm(mixed, weight, self.cfg["rms_norm_eps"]).astype(dtype)

    def observe(self, captures, cache):
        for row, block in enumerate(self.blocks):
            features = self.mix(captures, row, block.w["dflare_norm.weight"])
            block.observe(features, cache.entries[row])
        cache.feature = self.mix(
            [h[:, -1:] for h in captures], -1, self.markov["dflare_norm.weight"]
        )

    def fold(self, head):
        if self.folded is None:
            w, b = self.markov["hidden_out.weight"], self.markov["hidden_out.bias"]
            projection = mx.concatenate([w, b[:, None]], axis=1).astype(mx.float32)
            tiles = []
            for first in range(0, self.vocab, 4096):
                end = min(first + 4096, self.vocab)
                rows = (
                    head.rows(first, end)
                    if isinstance(head, AffineWeight)
                    else head[first:end]
                )
                tile = (rows.astype(mx.float32) @ projection).astype(w.dtype)
                mx.eval(tile)
                tiles.append(tile)
            self.folded = mx.concatenate(tiles)
            mx.eval(self.folded)
        return self.folded

    def propose(self, target, anchor, cache, count):
        """Propose greedy tokens from the Markov-conditioned draft logits."""
        if not 1 <= count < self.width:
            raise ValueError("DSpark proposal count exceeds its trained block")
        anchor_ids = mx.asarray(anchor, mx.int32).reshape(1)
        embed = target.embed_tokens(anchor_ids[None])
        dtype = embed.dtype
        masks = mx.broadcast_to(self.mask, (1, self.width - 1, embed.shape[-1]))
        h, residual = mx.concatenate([embed, masks], axis=1), None
        for i, block in enumerate(self.blocks):
            h, residual = block(h, residual, cache.entries[i])
        h = mx.fast.rms_norm(
            h.astype(mx.float32) + residual.astype(mx.float32),
            self.output_norm,
            self.cfg["rms_norm_eps"],
        ).astype(dtype)[:, 1 : count + 1]
        base = linear(h, target.unembed)[..., : self.vocab]
        rank = self.cfg["markov_rank"]
        weights = self.markov["rnn_proj.weight"].astype(mx.float32)
        inputs = mx.concatenate([h, mx.broadcast_to(cache.feature, h.shape)], axis=-1)
        conditioned = linear(
            inputs.astype(mx.float32), weights[:, 2 * rank :]
        ) + self.markov["rnn_proj.bias"].astype(mx.float32)
        folded = self.fold(target.unembed)
        state, prev = mx.zeros((1, rank), dtype), anchor_ids
        guesses = []
        for i in range(count):
            inputs = mx.concatenate(
                [state, self.markov["markov_w1.weight"][prev]], axis=-1
            )
            update = (
                linear(inputs.astype(mx.float32), weights[:, : 2 * rank])
                + conditioned[:, i]
            ).astype(dtype)
            gate, candidate, output = mx.split(update, 3, axis=-1)
            gate = mx.sigmoid(gate.astype(mx.float32)).astype(dtype)
            state = gate * state + (1 - gate) * mx.tanh(
                candidate.astype(mx.float32)
            ).astype(dtype)
            signal = mx.tanh(output.astype(mx.float32)).astype(dtype)
            bias = mx.addmm(folded[:, -1], signal, folded[:, :-1].T)
            scores = base[:, i] + bias
            prev = mx.argmax(scores, axis=-1)
            guesses.append(prev)
        return mx.stack(guesses, axis=-1)[0]
