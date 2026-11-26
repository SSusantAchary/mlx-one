# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, mtplx/models/inkling/model.py.
"""Inkling text trunk using MLX-One cache ownership and stock MLX projections."""

import os

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import ModelOutput
from mlx_one.engine.contracts import LayerExecutionSpec
from mlx_one.ops.logits import LogitsMode

from . import expert_scatter, grouped_experts, prompt_attention, short_conv, verify
from .activations import expert_inputs
from .attention import attend, chronological_attention
from .attention_metal import window_panel_attention
from .cache import InklingKVCache
from .long_attention import _supported_gpu, try_global_attention
from .precision import AffineWeight, packed_logical_shape


def _native_default(option):
    """Use the measured native route on the pinned M4, with explicit overrides."""
    value = os.environ.get("MLX_ONE_INKLING_" + option)
    if value is not None:
        return value == "1"
    return mx.default_device() == mx.gpu and _supported_gpu()


def linear(x, weight):
    if isinstance(weight, AffineWeight):
        if verify.active.get() is not None and x.ndim == 3 and x.shape[1] > 1:
            return verify.affine_panel(x, weight)
        return weight.product(x)
    if verify.active.get() is not None and x.ndim == 3 and x.shape[1] > 1:
        return verify.rowwise(lambda row: row @ weight.T, x)
    return x @ weight.T


def shared_linear(x, weights):
    return [linear(x, w) for w in weights]


class AttentionProjections(nn.Module):
    """One affine weight bank for the query, relative projection, and gate."""

    def __init__(self, weights):
        super().__init__()
        sizes = [w.codes.shape[0] for w in weights]
        self.cuts = [sizes[0], sizes[0] + sizes[1]]
        self.bank = AffineWeight(
            *(mx.concatenate([getattr(w, field) for w in weights], axis=0)
              for field in ("codes", "scales", "biases")),
            weights[0].bits, weights[0].group_size,
        )
        self.proofs = {}

    @classmethod
    def prepare(cls, weights):
        if not all(isinstance(w, AffineWeight) for w in weights):
            return None
        formats = {(w.bits, w.group_size, w.codes.shape[1], w.scales.dtype, w.biases.dtype)
                   for w in weights}
        if len(formats) != 1 or any(w.codes.shape[0] % 8 for w in weights):
            return None
        return cls(weights)

    def reference(self, x):
        bank = self.bank
        return shared_linear(x, [
            AffineWeight(bank.codes[start:stop], bank.scales[start:stop],
                         bank.biases[start:stop], bank.bits, bank.group_size)
            for start, stop in zip((0, *self.cuts), (*self.cuts, bank.codes.shape[0]), strict=True)
        ])

    def __call__(self, x):
        rows = x.shape[1]
        panel = verify.active.get() is not None
        if mx.default_device() != mx.gpu or rows not in (1, 2, 3, 4, 128):
            return self.reference(x)
        signature = (rows, panel, x.dtype)
        proof = self.proofs.get(signature)
        if proof is False:
            return self.reference(x)
        try:
            output = mx.split(linear(x, self.bank), self.cuts, axis=-1)
            if proof is None:
                reference = self.reference(x)
                self.proofs[signature] = bool(mx.all(mx.stack([
                    verify.same_bits(a, b) for a, b in zip(output, reference, strict=True)
                ])).item())
                if not self.proofs[signature]:
                    return reference
            return output
        except (RuntimeError, ValueError):
            self.proofs[signature] = False
            return self.reference(x)


def cross_pass_message(queries, states, scale):
    if verify.active.get() is not None and states.shape[1] > 1:
        return verify.rowwise(lambda row: cross_pass_message(queries, row, scale), states)
    return mx.softmax((queries @ states.swapaxes(-1, -2)) * scale, axis=-1) @ states


def norm(x, weight, eps):
    return mx.fast.rms_norm(x.astype(mx.float32), weight, eps)


def clip_heads(x, limit, scale=None):
    value = x.astype(mx.float32)
    if scale is not None:
        value = value * scale.astype(mx.float32)
    factor = mx.minimum(
        1.0, limit / (mx.linalg.norm(value, axis=-1, keepdims=True) + 1e-6)
    )
    return (value * factor).astype(x.dtype)


def read_blocks(closed, partial, queries, eps, normalize_values):
    values = mx.stack([*closed, partial], axis=-2).astype(mx.float32)
    keys = mx.fast.rms_norm(values, None, eps)
    q = queries.reshape(-1, partial.shape[-1]).astype(mx.float32)
    if verify.active.get() is not None and partial.shape[1] > 1:
        # MLX can change the dot product when the pass has several token rows.
        scores = verify.rowwise(lambda row: q @ row.swapaxes(-1, -2), keys)
    else:
        scores = q @ keys.swapaxes(-1, -2)
    p = mx.softmax(scores * partial.shape[-1] ** -0.5, axis=-1)
    source = keys if normalize_values else values
    if verify.active.get() is not None and partial.shape[1] > 1:
        return mx.concatenate([p[:, i:i + 1] @ source[:, i:i + 1]
                               for i in range(partial.shape[1])], axis=1)
    return p @ source


class Controller(nn.Module):
    def __init__(self, weights, stem):
        super().__init__()
        self.down = weights.pop(stem + ".down_weight")
        self.forget = weights.pop(stem + ".forget_weight").astype(mx.float32)
        self.gain = weights.pop(stem + ".gain_weight").astype(mx.float32)

    def __call__(self, x, message, eps):
        x = x.astype(mx.float32)
        message = message.astype(mx.float32)
        mixed = mx.concatenate([norm(x, None, eps), norm(message, None, eps)], axis=-1)
        state = linear(mixed.astype(self.down.dtype), self.down)
        state = state.astype(mx.float32)
        forget, gain = shared_linear(state, [self.forget, self.gain])
        return (1 - mx.tanh(forget)) * x + mx.tanh(gain) * message


class PackedExperts(nn.Module):
    """NVFP4 experts retain their packed weight and per-expert scale."""

    def __init__(self, weights, stem, mode):
        super().__init__()
        if mode != "nvfp4":
            raise ValueError("Inkling native text currently requires NVFP4 experts")
        self.codes = weights.pop(stem).view(mx.uint32)
        self.scales = weights.pop(stem + "_scale")
        self.outer = weights.pop(stem + "_amax") / weights.pop(stem + "_outer_scale_denominator")

    def __call__(self, rows, experts, row_ids=None, *, sorted_routes=False, grouped=None):
        inputs, row_scales = expert_inputs(rows, "nvfp4")

        def gather():
            return mx.gather_qmm(inputs, self.codes, scales=self.scales, rhs_indices=experts,
                                 lhs_indices=row_ids, group_size=16, bits=4, mode="nvfp4",
                                 sorted_indices=sorted_routes)

        if grouped is None:
            result = gather()
        else:
            signature = ("grouped_nvfp4", inputs.shape, self.codes.shape)
            result = verify.qualified(
                signature,
                lambda: grouped_experts.raw_product(inputs, self.codes, self.scales, *grouped),
                gather,
            )
        scale = row_scales if row_ids is None else row_scales[row_ids]
        return (result * (scale * self.outer[experts][..., None, None])).astype(rows.dtype)


@mx.compile
def activated_experts(pairs, gate_cap, up_cap):
    gate = pairs[..., 0::2].astype(mx.float32)
    up = pairs[..., 1::2].astype(mx.float32)
    return (
        gate_cap
        * mx.tanh(gate / gate_cap)
        * mx.sigmoid(gate)
        * (up_cap * mx.tanh(up / up_cap))
    ).astype(pairs.dtype)


class Experts(nn.Module):
    def __init__(self, weights, config, mode):
        super().__init__()
        self.router = weights.pop("mlp.gate.weight").astype(mx.float32)
        self.bias = weights.pop("mlp.gate.bias")
        self.gate_up = PackedExperts(weights, "mlp.experts.w13_weight", mode)
        self.down = PackedExperts(weights, "mlp.experts.w2_weight", mode)
        self.count = config["num_experts_per_tok"]
        self.scale = config["route_scale"]
        self.gate_cap = config["mlp_gate_softcap"]
        self.up_cap = config["mlp_up_softcap"]
        self.scatter_output = os.environ.get("MLX_ONE_INKLING_EXPERT_SCATTER") == "1"
        self.grouped_verify = os.environ.get("MLX_ONE_INKLING_GROUPED_EXPERTS") == "1"

    def __call__(self, hidden):
        if verify.active.get() is not None and hidden.shape[1] > 1:
            signature = ("experts", hidden.shape[1], hidden.dtype, self.count,
                         self.gate_up.codes.shape, self.down.codes.shape)
            return verify.qualified(signature, lambda: self._forward(hidden),
                                    lambda: verify.rowwise(self._forward, hidden))
        return self._forward(hidden)

    def _forward(self, hidden):
        flat = hidden.reshape(-1, hidden.shape[-1])
        router_input = flat.astype(mx.float32)
        if verify.active.get() is not None and flat.shape[0] > 1:
            logits = mx.concatenate([linear(row, self.router)
                                     for row in mx.split(router_input, flat.shape[0], axis=0)])
        else:
            logits = linear(router_input, self.router)
        scores = mx.sigmoid(logits)
        selected = mx.argpartition(-(scores + self.bias), kth=self.count - 1, axis=-1)[
            :, : self.count
        ]
        prob = mx.take_along_axis(scores, selected, axis=-1)
        prob = prob / (prob.sum(axis=-1, keepdims=True) + 1e-20) * self.scale
        experts = selected.flatten()
        if (self.grouped_verify and verify.active.get() is not None
                and 2 <= flat.shape[0] <= 4 and self.count == 8):
            out = self._grouped_panel(flat, experts)
        else:
            out = self._project_routes(flat, experts)
        out = out.reshape(*selected.shape, hidden.shape[-1])
        return (
            (out.astype(mx.float32) * prob[..., None])
            .sum(axis=-2)
            .astype(hidden.dtype)
            .reshape(hidden.shape)
        )

    def _grouped_panel(self, flat, experts):
        order = mx.argsort(experts)
        sorted_experts = experts[order]
        rows = mx.arange(experts.size, dtype=mx.uint32) // self.count
        sorted_rows = (order // self.count).astype(mx.uint32)
        # Both products return original router order, including when either
        # first-use check declines and the native gather supplies the result.
        pairs = self.gate_up(flat[:, None, :], experts, rows,
                             grouped=(sorted_experts, sorted_rows, order))
        mid = activated_experts(pairs, self.gate_cap, self.up_cap)
        return self.down(mid, experts, grouped=(sorted_experts, order, order))[:, 0, :]

    def _project_routes(self, flat, experts):
        group = flat.shape[0] > 32
        order = mx.argsort(experts) if group else mx.arange(experts.size)
        experts = experts[order] if group else experts
        pairs = self.gate_up(
            flat[:, None, :], experts, order // self.count, sorted_routes=group
        )
        mid = activated_experts(pairs, self.gate_cap, self.up_cap)
        out = self.down(mid, experts, sorted_routes=group)[:, 0, :]
        if group:
            restored = (expert_scatter.restore(out, order)
                        if self.scatter_output and flat.shape[0] <= 128 else None)
            out = restored if restored is not None else out[mx.argsort(order)]
        return out


class Convolution(nn.Module):
    def __init__(self, weights, stem, is_dynamic=False):
        super().__init__()
        self.dynamic = weights.pop(stem + ".weight_proj") if is_dynamic else None
        self.weight = (
            weights.pop(stem + ".static_w")
            if is_dynamic
            else weights.pop(stem + ".weight")[:, 0, :].T
        ).astype(mx.float32)
        self.fused_decode = _native_default("FUSED_CONV")
        self.prefill_conv_proofs = None

    def __call__(self, x, cache, slot, drive=None, add=None, out_dtype=None, tape=None):
        change = None if self.dynamic is None else linear(drive, self.dynamic)
        if tape is not None:
            tape.record_conv(slot, cache[slot], x, self.weight.shape[0] - 1)
        operands = (x, cache[slot], self.weight, change, add, out_dtype)
        if tape is not None:
            result = short_conv.verify_panel(*operands, fused_decode=self.fused_decode)
        elif x.shape[1] > 1 and self.prefill_conv_proofs is not None:
            result = short_conv.fused_prefill(*operands, proofs=self.prefill_conv_proofs)
        else:
            result = short_conv.fused(*operands) if self.fused_decode else None
        if result is None:
            result = short_conv.reference(*operands)
        out, cache[slot] = result
        return out


class Layer(nn.Module):
    def __init__(self, weights, cfg, index, attn_query, mlp_query, mode, *,
                 overlap_prefill=False, window_panel=False):
        super().__init__()
        self.cfg = cfg
        self.sliding = index in cfg["local_layer_ids"]
        self.long_attention = (
            not self.sliding and _native_default("LONG_ATTN")
        )
        self.flash_prompt = os.environ.get("MLX_ONE_INKLING_FLASH_PROMPT") == "1"
        self.overlap_prefill = overlap_prefill
        self.window_panel = window_panel
        self.boundary = index % cfg["n_layers_per_attn_res_block"] == 0
        self.attn_query, self.mlp_query = attn_query, mlp_query
        self.eps = cfg["rms_norm_eps"]
        self.qkv_norm = (
            [weights.pop("attn_norm.weight")]
            if self.sliding
            else [weights.pop(k + "_input_norm.weight") for k in ("q", "k", "v")]
        )
        self.ffn_norm = weights.pop("mlp_norm.weight")
        stem = "attn."
        self.q, self.k, self.v, self.r, self.sink, self.gate, self.out = (
            weights.pop(stem + k + ".weight")
            for k in (
                "wq_du",
                "wk_dv",
                "wv_dv",
                "wr_du",
                "attn_sink_proj",
                "gate_attn",
                "wo_ud",
            )
        )
        self.q_scale = weights.pop(stem + "q_l2_clip.scale")
        self.r_scale = weights.pop(stem + "rel_l2_clip.scale")
        self.rel_basis = weights.pop(stem + "rel_logits_proj.proj")
        self.attention_projections = None
        if os.environ.get("MLX_ONE_INKLING_PROJECTION_BANK") == "1":
            self.attention_projections = AttentionProjections.prepare((self.q, self.r, self.gate))
            if self.attention_projections is not None:
                self.q = self.r = self.gate = None
        self.convs = [
            Convolution(weights, stem + k, not self.sliding)
            for k in ("k_sconv", "v_sconv")
        ]
        self.convs += [
            Convolution(weights, "attn_sconv"),
            Convolution(weights, "mlp_sconv"),
        ]
        self.experts = Experts(weights, cfg, mode)
        self.controllers = (
            [Controller(weights, "controllers." + k) for k in ("attn", "k", "v", "mlp")]
            if "controllers.attn.down_weight" in weights
            else []
        )
        if weights:
            raise ValueError(
                f"Unrecognized Inkling layer {index} tensors: {sorted(weights)}"
            )

    def attention(self, q_input, k_input, v_input, kv, conv, tape=None):
        cfg = self.cfg
        shape = (*q_input.shape[:2], self.q_scale.shape[0])
        width = self.q_scale.shape[-1]
        if self.attention_projections is not None:
            q_lin, r_lin, gate_lin = self.attention_projections(q_input)
            sink = linear(q_input, self.sink)
            k_lin, v_lin = linear(k_input, self.k), linear(v_input, self.v)
        elif q_input is k_input and k_input is v_input:
            q_lin, r_lin, sink, gate_lin, k_lin, v_lin = shared_linear(
                q_input, [self.q, self.r, self.sink, self.gate, self.k, self.v],
            )
        else:
            q_lin, r_lin, sink, gate_lin = shared_linear(
                q_input, [self.q, self.r, self.sink, self.gate]
            )
            k_lin = linear(k_input, self.k)
            v_lin = linear(v_input, self.v)
        k_conv = self.convs[0](k_lin, conv, 0, k_input, tape=tape)
        v_conv = self.convs[1](v_lin, conv, 1, v_input, tape=tape)
        q = clip_heads(q_lin.reshape(*shape, width), cfg["qk_l2_clip_threshold"], self.q_scale)
        rel = clip_heads(r_lin.reshape(*shape, -1), cfg["rel_l2_clip_threshold"], self.r_scale)
        k = clip_heads(k_conv.reshape(*k_conv.shape[:2], -1, width), cfg["qk_l2_clip_threshold"])
        v = v_conv.reshape(*v_conv.shape[:2], -1, width)
        if cfg["use_value_norm_no_gamma"]:
            v = norm(v, None, self.eps).astype(v_input.dtype)
        if not self.sliding:
            pos = mx.arange(q_input.shape[1]) + kv.offset + 1
            tau = 1 + cfg["log_scaling_alpha"] * mx.log(
                mx.maximum(pos / cfg["log_scaling_n_floor"], 1)
            )
            q = (q.astype(mx.float32) * tau[:, None, None]).astype(q.dtype)
            rel = (rel.astype(mx.float32) * tau[:, None, None]).astype(rel.dtype)
        window = cfg["sliding_window_size"] if self.sliding else 0
        query = q.transpose(0, 2, 1, 3)
        k_rows, v_rows = k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        basis = self.rel_basis.astype(rel.dtype)
        if tape is not None:
            profiles = mx.concatenate(
                [row @ basis for row in mx.split(rel, rel.shape[1], 1)], axis=1,
            )
        else:
            profiles = rel @ basis
        k, v = kv.update(k_rows, v_rows)
        heads_first = None
        if tape is not None:
            if window:
                tape.window = (k, v)
                if self.window_panel and mx.default_device() == mx.gpu:
                    signature = ("window_attention", query.shape[2], window, k.shape[2] >= window)
                    heads_first = verify.qualified(
                        signature,
                        lambda: window_panel_attention(query, k, v, profiles, sink, window),
                        lambda: chronological_attention(query, k, v, profiles, sink, window),
                    )
            elif hasattr(kv, "panel_attention"):
                heads_first = kv.panel_attention(query, k_rows, v_rows, profiles, sink)
            else:
                heads_first = try_global_attention(
                    query, kv.keys, kv.values, profiles, sink, kv.offset, verify=True,
                )
            if heads_first is None:
                heads_first = chronological_attention(query, k, v, profiles, sink, window)
        elif self.flash_prompt and q_input.shape[1] > 32:
            if window:
                heads_first = prompt_attention.prompt_attention(
                    query, k, v, profiles, sink, window,
                )
            else:
                heads_first = prompt_attention.prompt_attention(
                    query, kv.keys, kv.values, profiles, sink, window, count=kv.offset,
                )
        if heads_first is None and self.long_attention:
            heads_first = try_global_attention(query, kv.keys, kv.values, profiles, sink, kv.offset)
        if heads_first is None:
            if self.overlap_prefill and tape is None and q_input.shape[1] > 1:
                heads_first = attend(query, k, v, profiles, sink, window, overlap_tiles=True)
            else:
                heads_first = attend(query, k, v, profiles, sink, window)
        h = heads_first.transpose(0, 2, 1, 3).reshape(*q_input.shape[:2], -1)
        gated = (h.astype(mx.float32) * mx.sigmoid(gate_lin.astype(mx.float32))).astype(h.dtype)
        return linear(gated, self.out)

    def __call__(self, x, closed, kv, conv, messages=None, tape=None):
        read = (read_blocks(closed, x, self.attn_query, self.eps,
                            self.cfg["attn_res_normalize_values"])
                if self.attn_query is not None else x[..., None, :])
        if self.sliding and messages is None:
            q = k = v = norm(read[..., 0, :], self.qkv_norm[0], self.eps).astype(x.dtype)
        else:
            sources = [read[..., 0 if self.sliding else i, :] for i in range(3)]
            if messages is not None:
                sources = [self.controllers[i](h, messages[..., i, :], self.eps)
                           for i, h in enumerate(sources)]
            q, k, v = [norm(h, self.qkv_norm[0 if self.sliding else i], self.eps).astype(x.dtype)
                       for i, h in enumerate(sources)]
        attn = self.attention(q, k, v, kv, conv, tape)
        residual = self.convs[2](
            attn, conv, 2, add=None if self.boundary else x, out_dtype=mx.float32, tape=tape,
        )
        if self.boundary:
            closed.append(x)
        ffn = read_blocks(closed, residual, self.mlp_query, self.eps,
                          self.cfg["attn_res_normalize_values"])[..., 0, :]
        if messages is not None:
            ffn = self.controllers[3](ffn, messages[..., 3, :], self.eps)
        routed = self.experts(norm(ffn, self.ffn_norm, self.eps).astype(x.dtype))
        return self.convs[3](routed, conv, 3, add=residual, out_dtype=x.dtype, tape=tape)


class Embedding(nn.Module):
    def __init__(self, weight, gain, eps):
        super().__init__()
        self.gain, self.eps = gain, eps
        # A 4-bit embedding (Bare Speed) dequantizes only the rows a step gathers; `weight` then
        # answers .dtype and .shape for the code that sizes buffers from it, and holds no data.
        self.packed = weight if isinstance(weight, AffineWeight) else None
        self.weight = packed_logical_shape(weight) if self.packed is not None else weight

    def __call__(self, ids):
        if self.packed is None:
            return mx.fast.rms_norm(self.weight[ids], self.gain, self.eps)
        w = self.packed
        rows = mx.dequantize(
            w.codes[ids], w.scales[ids], w.biases[ids], bits=w.bits, group_size=w.group_size
        )
        return mx.fast.rms_norm(rows, self.gain, self.eps)


class InklingForCausalLM(nn.Module):
    def __init__(self, config, weights, quant_modes):
        super().__init__()
        self.config = config
        cfg = self.cfg = config.text
        loop = cfg["loop_config"]
        self.start, self.end, self.passes = (
            loop["core_start"],
            loop["core_end"],
            loop["num_passes"],
        )
        count = cfg["num_hidden_layers"]
        self.visit_order = config.visit_order
        self.eps = cfg["rms_norm_eps"]
        self.verify_panels = _native_default("VERIFY_PANEL")
        self.continuous_dspark = (
            self.verify_panels and _native_default("CONTINUOUS_DSPARK")
        )
        self.replay_mode = os.environ.get("MLX_ONE_INKLING_METAL_REPLAY", "")
        self.replay_extension = None
        self.replay_error = None
        if self.continuous_dspark and self.replay_mode in {"panel", "round"} and _supported_gpu():
            from mlx_one.mlxboost.native.capabilities import load_extension

            try:
                extension = load_extension()
                extension.install_replay()
                self.replay_extension = extension
            except (RuntimeError, AttributeError) as exc:
                self.replay_error = str(exc)
        self.adaptive_verification = (
            self.verify_panels and os.environ.get("MLX_ONE_INKLING_ADAPTIVE_DSPARK") == "1"
        )
        self.commit_panel_prefix = os.environ.get("MLX_ONE_INKLING_PANEL_PREFIX_COMMIT") == "1"
        self.async_draft_commit = os.environ.get("MLX_ONE_INKLING_ASYNC_DRAFT_COMMIT") == "1"
        self.verify_panel_proofs = {}
        self.verify_stage_proofs = {}
        overlap_prefill = _native_default("PREFILL_OVERLAP")
        self.overlap_prefill = (
            overlap_prefill and mx.default_device() == mx.gpu and _supported_gpu()
        )
        self.group_prefill = (self.overlap_prefill
                              and os.environ.get("MLX_ONE_INKLING_PREFILL_GROUP_512") == "1")
        window_panel = (self.verify_panels and os.environ.get("MLX_ONE_INKLING_WINDOW_PANEL") == "1"
                        and mx.default_device() == mx.gpu and _supported_gpu())
        self.embed_tokens = Embedding(
            weights.pop("embed.weight"), weights.pop("embed_norm.weight"), self.eps
        )
        self.final_norm, self.unembed = (
            weights.pop("norm.weight"),
            weights.pop("unembed.weight"),
        )
        self.final_query = weights.pop("attn_res.output_query")
        size = self.end - self.start
        main = weights.pop("middle_cycle_cross_pass_queries").reshape(size, 2, -1)
        kv = weights.pop("middle_cycle_cross_pass_kv_queries").reshape(size, 2, -1)
        self.message_queries = mx.concatenate([main[:, :1], kv, main[:, 1:]], axis=1)
        self.capture_ids = tuple(config.draft["target_layer_ids"])
        self.layers = []
        for index in range(count):
            stem = f"layers.{index}."
            params = {
                key[len(stem) :]: weights.pop(key)
                for key in tuple(weights)
                if key.startswith(stem)
            }
            aq = weights.pop(f"attn_res.attn_queries.{index - 1}") if index else None
            mq = weights.pop(f"attn_res.mlp_queries.{index}")
            self.layers.append(Layer(
                params, cfg, index, aq, mq, quant_modes[index],
                overlap_prefill=self.overlap_prefill, window_panel=window_panel,
            ))
        if weights:
            raise ValueError(f"Unrecognized Inkling target tensors: {sorted(weights)}")
        self.prefill_conv_proofs = {}
        if os.environ.get("MLX_ONE_INKLING_FUSED_PREFILL_CONV") == "1":
            for layer in self.layers:
                for conv in layer.convs:
                    conv.prefill_conv_proofs = self.prefill_conv_proofs

    def make_cache(self):
        return tuple(InklingKVCache(self.cfg["sliding_window_size"]
                                    if self.layers[index].sliding else 0)
                     for index in self.visit_order)

    def prefill_width(self, maximum, prompt_tokens):
        return min(maximum, 512 if self.group_prefill else 128)

    def prefill_group(self, ids, *, cache, logits_mode, output_hidden_states=False,
                      cancelled):
        # Keep every projection, attention and draft observation at the B1
        # reference width. Only the submission and synchronization change.
        hidden, pending = [], []
        for first in range(0, ids.shape[1], 128):
            if cancelled():
                mx.synchronize()
                return None
            tile, captures = self.forward(ids[:, first:first + 128], cache, output_hidden_states)
            state = [t for entry in cache for t in entry.tensors()]
            if output_hidden_states:
                draft = cache[0].draft
                self.drafter.observe(captures, draft)
                state.extend(draft.tensors())
            tile = mx.depends(tile, state)
            mx.async_eval(tile)
            del state
            # The GPU submission owns its inputs. Keep only the completion token
            # on the host so append-only KV banks can be donated to the next tile.
            hidden.append(tile)
            pending.append(tile)
            if len(pending) >= 2:
                mx.eval(pending.pop(0))
        last = mx.concatenate(hidden, axis=1)
        logits = None if logits_mode == LogitsMode.NO_LOGITS else self.head(hidden[-1][:, -1:])
        return ModelOutput(logits, last, cache)

    def layer_execution_specs(self):
        cfg = self.cfg
        return tuple(
            LayerExecutionSpec(
                layer_id=visit, state_kind="sliding_kv" if window else "full_kv",
                query_heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"],
                key_dim=cfg["head_dim"], value_dim=cfg["head_dim"],
                position_policy="relative_bias", window_tokens=window,
            )
            for visit, index in enumerate(self.visit_order)
            for window in (cfg["sliding_window_size"] if self.layers[index].sliding else None,)
        )

    def _cache_capacity(self, tokens):
        if not 0 <= tokens <= self.config.max_position_embeddings:
            raise ValueError("Inkling request exceeds the checkpoint context")
        return ((tokens + 255) // 256) * 256

    def request_cache_bytes(self, tokens, *, speculative=False):
        capacity = self._cache_capacity(tokens)
        if not capacity:
            return 0
        cfg = self.cfg
        row_bytes = 4 * cfg["num_key_value_heads"] * cfg["head_dim"]
        tail = 4 * (cfg["sconv_kernel_size"] - 1) * (
            2 * cfg["hidden_size"] + 2 * cfg["num_key_value_heads"] * cfg["head_dim"])
        local_capacity = min(capacity, cfg["sliding_window_size"] + 256)
        amount = sum(row_bytes * (local_capacity if self.layers[index].sliding else capacity)
                     + tail for index in self.visit_order)
        return amount + (self.draft_cache_budget(capacity) if speculative else 0)

    def configure_request_cache(self, cache, tokens, *, speculative=False):
        """Reserve full attention banks once and bound the local visits by the window."""
        amount = self.request_cache_bytes(tokens, speculative=speculative)
        capacity = self._cache_capacity(tokens)
        if speculative:
            cache[0].draft = self.drafter.make_cache(capacity)
        for entry in cache:
            if capacity and not entry.max_size and entry.kv.keys is None:
                entry.kv.step = capacity
        return amount

    def draft_cache_budget(self, capacity):
        drafter = getattr(self, "drafter", None)
        if drafter is None:
            raise ValueError("Inkling MTP requires the native DSpark sidecar")
        cfg = self.config.draft
        row_bytes = 4 * cfg["num_key_value_heads"] * cfg["head_dim"]
        return sum(row_bytes * (min(capacity, b.window + 128) if b.window else capacity)
                   for b in drafter.blocks) + 2 * cfg["hidden_size"]

    def head(self, h):
        cap = self.cfg["final_logit_softcapping"]
        logits = linear(h.astype(self.embed_tokens.weight.dtype), self.unembed)
        logits = logits[..., : self.cfg["unpadded_vocab_size"]]
        return mx.tanh(logits / cap) * cap if cap else logits

    def forward(self, ids, cache, capture=False, tapes=None):
        if ids.ndim != 2 or ids.shape[0] != 1 or len(cache) != len(self.visit_order):
            raise ValueError("Inkling text requires B1 and one cache per loop visit")
        closed, captured = [], {}
        x = self.embed_tokens(ids)
        prefill = tapes is None and ids.shape[1] > 1
        overlap = prefill and self.overlap_prefill
        pending = []
        for visit, index in enumerate(self.visit_order):
            if index == self.start:
                if visit == self.start:
                    entry, entry_blocks = x, list(closed)
                    messages = None
                else:
                    prior = [*closed[len(entry_blocks) + 1 :], x]
                    states = mx.stack(prior, axis=-2).astype(mx.float32)
                    messages = mx.fast.rms_norm(states, None, self.eps)
                    x, closed = entry, list(entry_blocks)
            if capture and visit in self.capture_ids:
                captured[visit] = x
            message = None
            if self.start <= index < self.end and messages is not None:
                message = cross_pass_message(
                    self.message_queries[index - self.start], messages, x.shape[-1] ** -0.5
                )
            state = cache[visit]
            tape = tapes[visit] if tapes is not None else None
            x = self.layers[index](x, closed, state.kv, state.conv, message, tape)
            if overlap and (visit + 1) % 4 == 0:
                # Bound evaluated work to two groups of visits, as in the tiled read.
                checkpoint = (x, *state.tensors())
                mx.async_eval(*checkpoint)
                if len(pending) == 2:
                    mx.eval(*pending.pop(0))
                pending.append(checkpoint)
            elif prefill and not overlap and not state.max_size:
                # Finish each global attention before building the next long-context mask.
                mx.eval(x, *state.tensors())
        last = read_blocks(closed, x, self.final_query, self.eps,
                           self.cfg["attn_res_normalize_values"])[..., 0, :]
        hidden = norm(last, self.final_norm, self.eps).astype(x.dtype)
        if capture and len(self.visit_order) in self.capture_ids:
            captured[len(self.visit_order)] = hidden
        return hidden, tuple(captured[i] for i in self.capture_ids) if capture else None

    def verify_panel(self, ids, cache):
        tapes = [verify.VisitTape(ids.shape[1]) for _ in cache]
        token = verify.active.set(self.verify_stage_proofs)
        try:
            hidden, captures = self.forward(ids, cache, capture=True, tapes=tapes)
            return self.head(hidden), captures, tapes
        finally:
            verify.active.reset(token)

    def __call__(self, input_ids, *, cache=None, logits_mode=None, output_hidden_states=False):
        cache = self.make_cache() if cache is None else cache
        hidden, captures = self.forward(input_ids, cache, capture=output_hidden_states)
        logits = None
        if logits_mode != LogitsMode.NO_LOGITS:
            logits = self.head(hidden[:, -1:] if logits_mode == LogitsMode.LAST_ONLY else hidden)
        return ModelOutput(logits, hidden, cache, hidden_states=captures)
