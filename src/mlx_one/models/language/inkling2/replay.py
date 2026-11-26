"""Fixed four-row, read-only DSpark programs for MLX-Boost's Metal recorder."""

import os
from types import SimpleNamespace

import mlx.core as mx

from ._replay_draft_attention import draft_attention
from .long_attention import _launch
from .verify import VisitTape, same_bits

ROWS = 4
WORKSPACE_LIMIT = 512 << 20


def packet_for(logits, guesses):
    predictions = mx.argmax(logits[0], axis=-1).astype(mx.int32)
    accepted = mx.cumprod((predictions[:-1] == guesses).astype(mx.int32)).sum()
    return mx.concatenate([accepted.reshape(1), predictions])


class PanelKV:
    """Expose the same chronological window and hold global writes as four-row deltas."""

    def __init__(self, keys, values, position, window, long):
        self.keys, self.values = keys, values
        self.offset, self.window, self.long = position, window, long

    def update(self, keys, values):
        self.appended = (keys, values)
        if self.window:
            return (mx.concatenate([self.keys, keys], axis=2),
                    mx.concatenate([self.values, values], axis=2))
        return keys, values

    def panel_attention(self, query, key, value, profiles, sink):
        if self.long:
            return _launch(query, self.keys, self.values, profiles, sink, self.offset + ROWS,
                           ((0, ROWS, 32),), True, appended=(mx.contiguous(key),
                                                            mx.contiguous(value)))
        # Below 4096 MLX uses one-pass vector attention. Preserve its per-row
        # reduction order and mask the unused suffix of the fixed-size bank.
        keys = mx.slice_update(self.keys, key, self.offset, (2,))
        values = mx.slice_update(self.values, value, self.offset, (2,))
        columns = mx.arange(keys.shape[2])
        parts = []
        for row in range(ROWS):
            distance = self.offset + row - columns
            profile = profiles[:, row, :, :]
            extent = profile.shape[-1]
            bias = mx.take(profile, mx.clip(distance, 0, extent - 1), axis=-1)
            bias = mx.where(distance >= extent, 0, bias)
            bias = mx.where(distance >= 0, bias, -float("inf")).astype(query.dtype)
            parts.append(mx.fast.scaled_dot_product_attention(
                query[:, :, row:row + 1], keys, values, scale=query.shape[-1] ** -0.5,
                mask=bias[:, :, None], sinks=sink[0, row],
            ))
        return mx.concatenate(parts, axis=2)


class DraftKV:
    def __init__(self, keys, values, position, window, long):
        self.keys, self.values = keys, values
        self.offset, self.window, self.long = position, window, long

    def attention(self, query, keys, values):
        if self.long and not self.window:
            return draft_attention(query, self.keys, self.values, keys, values, self.offset)
        key, value, valid = self.proposal_kv(keys, values)
        return mx.fast.scaled_dot_product_attention(
            query, key, value, scale=query.shape[-1] ** -0.5, mask=valid,
        )

    def proposal_kv(self, keys, values):
        joined = [mx.concatenate([self.keys, keys], axis=2),
                  mx.concatenate([self.values, values], axis=2)]
        if self.window:
            return *joined, None
        # Place the proposal immediately after the live history so the
        # attention's reduction indexes the same keys as the ordinary drafter.
        key, value = [mx.slice_update(bank, update, self.offset, (2,))
                      for bank, update in zip(joined, (keys, values), strict=True)]
        valid = mx.arange(key.shape[2]) < self.offset + keys.shape[2]
        return key, value, valid[None, None, None, :]


def fixed_program(model, inputs, windows, draft_windows, long, full):
    token, position, *state = inputs
    leaves = iter(state)
    cache = []
    for window in windows:
        kv = PanelKV(next(leaves), next(leaves), position, window, long)
        cache.append(SimpleNamespace(kv=kv, conv=[next(leaves) for _ in range(4)]))
    if full:
        draft = SimpleNamespace(entries=[DraftKV(next(leaves), next(leaves), position, w, long)
                                         for w in draft_windows], feature=next(leaves))
        guesses = model.drafter.propose(model, token, draft, ROWS - 1).astype(mx.int32)
        ids = mx.concatenate([token, guesses])
    else:
        ids = token
    logits, captures, tapes = model.verify_panel(ids[None], cache)
    return [ids, packet_for(logits, ids[1:]), logits, *captures,
            *(t for entry, tape in zip(cache, tapes, strict=True)
              for t in (*entry.kv.appended, *tape.conv_histories))]


def model_weights(tree):
    # A Module is a dict. Only evaluated array leaves can be pinned as weights.
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, dict):
        for key, child in tree.items():
            if isinstance(key, str) and not key.startswith("_"):
                yield from model_weights(child)
    elif isinstance(tree, (list, tuple)):
        for child in tree:
            yield from model_weights(child)


def _inputs(cache, token, full, long):
    offset = cache[0].offset
    # Views of global target banks can retain their native head stride. Small
    # sliding/conv views change strides after partial commits; normalize them.
    bound = min(4095, ((offset + ROWS + 255) // 256) * 256)
    inputs = [token, mx.array([offset], mx.int32)]
    for entry in cache:
        k, v = entry.kv.keys, entry.kv.values
        if entry.max_size:
            k, v = mx.contiguous(k), mx.contiguous(v)
        elif not long:
            k, v = k[:, :, :bound], v[:, :, :bound]
        inputs.extend((k, v, *(mx.contiguous(t) for t in entry.conv)))
    if full:
        draft = cache[0].draft
        for entry in draft.entries:
            k, v = entry.keys, entry.values
            if getattr(entry, "max_size", 0):
                k, v = mx.contiguous(k), mx.contiguous(v)
            inputs.extend((k, v))
        inputs.append(mx.contiguous(draft.feature))
    mx.eval(inputs)
    return inputs


def expected_outputs(cache, ids, logits, captures, tapes):
    return [ids, packet_for(logits, ids[1:]), logits, *captures,
            *(t for entry, tape in zip(cache, tapes, strict=True)
              for t in (*(x[:, :, -ROWS:] for x in (
                  tape.window if entry.max_size else entry.kv.state())), *tape.conv_histories))]


def install(cache, outputs, capture_count):
    """Publish provisional rows through the existing proposal transaction."""
    _, packet, logits, *state = outputs
    captures = tuple(state[:capture_count])
    leaves = iter(state[capture_count:])
    tapes = []
    for entry in cache:
        tape = VisitTape(ROWS)
        window = entry.kv.update(next(leaves), next(leaves))
        if entry.max_size:
            tape.window = window
        tape.conv_histories = [next(leaves) for _ in range(4)]
        entry.conv = [t[:, ROWS:] for t in tape.conv_histories]
        tapes.append(tape)
    return packet, logits, captures, tapes


class RoundReplay:
    """One live program per request. Qualify the target before recording the full round."""

    def __init__(self, extension, mode):
        self.extension = extension
        self.mode = mode
        self.geometry = self.program = self.reason = None
        self.stage = "panel"
        self.proven = False
        self.last_offset = None
        self.checks = 0

    @property
    def workspace_bytes(self):
        return self.program.workspace_bytes if self.program is not None else 0

    def decline(self, reason):
        self.reason = str(reason)
        self.program = None
        self.proven = False

    def plan(self, model, cache):
        offset = cache[0].offset
        windows = tuple(c.max_size for c in cache)
        draft_windows = tuple(b.window or 0 for b in model.drafter.blocks)
        if (offset < max((*windows, *draft_windows)) or offset < 4095 <= offset + ROWS - 1
                or any(c.offset != offset for c in (*cache, *cache[0].draft.entries))):
            return False
        long = offset >= 4095
        if long and os.environ.get("MLX_SDPA_BLOCKS"):
            return False
        geometry = (long, offset.bit_length(), None if long else (offset + ROWS + 255) // 256)
        if geometry != self.geometry:
            self.geometry = geometry
            self.stage, self.program, self.proven = "panel", None, False
            self.checks, self.last_offset, self.reason = 0, None, None
        return self.reason is None

    def prepare(self, proposal, first, draft_guess):
        model, cache = proposal.bundle.model, proposal.cache
        windows = tuple(c.max_size for c in cache)
        draft_windows = tuple(b.window or 0 for b in model.drafter.blocks)
        long = self.geometry[0]
        full = self.stage == "round"
        token = mx.array([first], mx.int32)
        if not full:
            token = mx.concatenate([token, draft_guess().astype(mx.int32)])
        inputs = _inputs(cache, token, full, long)
        if self.program is None:
            weights = list(model_weights(model))
            mx.eval(weights)
            # Build the lazy graph first so first-use stage probes and lazy weights
            # cannot introduce an unrelated command into the recording.
            outputs = fixed_program(model, inputs, windows, draft_windows, long, full)
            mx.synchronize()
            self.extension.begin_replay_capture()
            try:
                mx.eval(outputs)
                self.program = self.extension.finish_replay_capture(
                    inputs, outputs, weights, WORKSPACE_LIMIT,
                )
            finally:
                self.extension.abort_replay_capture()
        outputs = self.program.run(inputs)
        return outputs

    def check(self, candidate, reference, offset):
        if len(candidate) != len(reference):
            self.decline("output count changed")
            return
        checks = mx.stack([same_bits(a, b) for a, b in zip(candidate, reference, strict=True)])
        for index, equal in enumerate(checks.tolist()):
            if not equal:
                self.decline(f"{self.stage} output {index} differs at {offset}")
                return
        if offset == self.last_offset:
            return
        self.last_offset = offset
        self.checks += 1
        if self.checks >= 2:
            self.proven = True

    def advance(self):
        if self.stage == "panel" and self.mode == "round":
            self.stage = "round"
            self.program, self.proven, self.checks, self.last_offset = None, False, 0, None
