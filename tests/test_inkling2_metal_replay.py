from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_one.mlxboost.native.capabilities import load_extension
from mlx_one.mlxboost.native.schemas import NativeKernelUnavailable
from mlx_one.models.language.inkling2 import replay
from mlx_one.models.language.inkling2.attention import chronological_attention
from mlx_one.models.language.inkling2.cache import InklingKVCache
from mlx_one.models.language.inkling2.long_attention import _supported_gpu
from mlx_one.models.language.inkling2.verify import VisitTape, same_bits


@pytest.fixture
def metal():
    try:
        ext = load_extension()
    except NativeKernelUnavailable as exc:
        pytest.skip(str(exc))
    ext.install_replay()
    return ext


def test_metal_rebinds_views_keeps_prior_output_and_rejects_other_shapes(metal):
    def program(x):
        y = x * x + 1
        return [y, y[2:9:2]]

    x = mx.arange(36, dtype=mx.float32)[1:17]
    mx.eval(x)
    mx.synchronize()
    metal.begin_replay_capture()
    try:
        output = program(x)
        mx.eval(output)
        p = metal.finish_replay_capture([x], output, [], 65536)
    finally:
        metal.abort_replay_capture()
    assert p.dispatches > 0
    prior = None
    for start in (2, 4):
        z = mx.arange(36, dtype=mx.float32)[start:start + 16]
        mx.eval(z)
        got = p.run([z])
        assert all(same_bits(a, b).item() for a, b in zip(got, program(z), strict=True))
        if prior is None:
            prior = got
    assert prior[1].tolist() == [17, 37, 65, 101]
    with pytest.raises(RuntimeError, match="layout"):
        p.run([mx.array([0])])


@pytest.mark.parametrize("offset", [1026, 4091, 4095, 4129])
def test_global_replay_reads_only_live_history_and_panel(offset):
    if not _supported_gpu():
        pytest.skip("The native attention reduction is pinned to the base M4")
    rows = 4
    long = offset >= 4095
    def draw(shape, seed):
        return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)

    q = draw((1, 24, rows, 128), 1)
    p, s = draw((1, rows, 24, 1024), 2), draw((1, rows, 24), 3)
    bank_size = offset + 64 if long else min(offset + 64, 4095)
    banks = [draw((1, 8, bank_size, 128), n) for n in (4, 5)]
    new = [draw((1, 8, rows, 128), n) for n in (6, 7)]
    merged = [mx.concatenate([bank[:, :, :offset], tail], axis=2)
              for bank, tail in zip(banks, new, strict=True)]
    kv = replay.PanelKV(*banks, mx.array([offset], mx.int32), 0, long)
    actual = kv.panel_attention(q, *new, p, s)
    want = chronological_attention(q, *merged, p, s, 0)
    assert same_bits(actual, want).item()


def test_replay_panel_then_round_with_changing_anchor_and_partial_commits(metal):
    class Drafter:
        blocks = ()

        def propose(self, model, anchor, draft, count):
            return (mx.asarray(anchor) + mx.arange(1, count + 1) + draft.feature).flatten()

    class Target:
        drafter = Drafter()
        capture_ids = (0,)

        def verify_panel(self, ids, cache):
            tapes = []
            for e in cache:
                kv = ids.reshape(1, 1, -1, 1)
                window = e.kv.update(kv, kv)
                tape = VisitTape(4)
                if getattr(e.kv, "window", getattr(e, "max_size", 0)):
                    tape.window = window
                for i, tail in enumerate(e.conv):
                    tape.record_conv(i, tail, ids[..., None], 2)
                    e.conv[i] = tape.conv_histories[i][:, -2:]
                tapes.append(tape)
            logits = (mx.arange(64) == (ids + 1)[..., None]).astype(mx.float32)
            return logits, (ids + 1,), tapes

    cache = (InklingKVCache(2),)
    e = cache[0]
    hist = mx.array([1, 2]).reshape(1, 1, 2, 1)
    e.kv.update(hist, hist)
    e.kv._offset = 32
    e.conv = [mx.asarray(hist.reshape(1, 2, 1), mx.float32) + i for i in range(4)]
    e.draft = SimpleNamespace(entries=[], feature=mx.array([0]))
    model = Target()
    proposal = SimpleNamespace(bundle=SimpleNamespace(model=model), cache=cache)
    engine = replay.RoundReplay(metal, "round")
    served = []
    for i in range(8):
        anchor, offset = i + 3, e.offset
        def guesses(anchor=anchor):
            return model.drafter.propose(model, anchor, e.draft, 3)
        ids = mx.concatenate([mx.array([anchor]), guesses()])
        assert engine.plan(model, cache)
        candidate = engine.prepare(proposal, anchor, guesses)
        assert candidate is not None
        if engine.proven:
            served.append(engine.stage)
            packet, _, _, tapes = replay.install(cache, candidate, 1)
            assert packet.tolist()[0] == 3
            engine.advance()
        else:
            logits, captures, tapes = model.verify_panel(ids[None], cache)
            engine.check(candidate, replay.expected_outputs(cache, ids, logits, captures, tapes),
                         offset)
            assert engine.reason is None
        kept = 1 + i % 3
        e.restore(tapes[0].boundary(offset, kept, 2))
        mx.eval(*e.tensors())
        assert e.offset == offset + kept
        assert e.conv[0][0, -1, 0].item() == anchor + kept - 1
    assert "panel" in served and "round" in served


def test_prefill_group_uses_reference_tiles_and_observes_each_draft_prefix():
    from mlx_one.models.language.inkling2.model import InklingForCausalLM
    from mlx_one.ops.logits import LogitsMode

    entry = InklingKVCache()
    entry.draft = SimpleNamespace(feature=None)
    entry.draft.tensors = lambda: [entry.draft.feature]
    widths = []
    def forward(ids, cache, capture):
        widths.append(ids.shape[1])
        rows = ids.reshape(1, 1, -1, 1)
        cache[0].kv.update(rows, rows)
        return ids[..., None], (ids,)

    def observe(captures, draft):
        last = captures[0][0, -1:]
        draft.feature = last if draft.feature is None else mx.concatenate([draft.feature, last])

    target = SimpleNamespace(forward=forward, drafter=SimpleNamespace(observe=observe),
                             head=lambda x: x)
    ids = mx.arange(493)[None]
    output = InklingForCausalLM.prefill_group(target, ids, cache=(entry,),
                                              logits_mode=LogitsMode.LAST_ONLY,
                                              output_hidden_states=True, cancelled=lambda: False)
    assert widths == [128, 128, 128, 109]
    assert same_bits(output.last_hidden_state[..., 0], ids).item()
    assert output.logits.item() == 492 and entry.offset == 493
    assert entry.draft.feature.tolist() == [127, 255, 383, 492]


def test_global_draft_ignores_unused_capacity_and_tracks_position():
    from mlx_one.models.language.inkling2._replay_draft_attention import draft_attention

    q = mx.random.normal((1, 24, 8, 128), key=mx.random.key(7)).astype(mx.bfloat16)
    own = [mx.random.normal((1, 8, 8, 128), key=mx.random.key(i)).astype(q.dtype)
           for i in (8, 9)]
    bank = [mx.random.normal((1, 8, 768, 128), key=mx.random.key(i)).astype(q.dtype)
            for i in (10, 11)]
    for count in (429, 660):
        def run(banks, count=count):
            return draft_attention(q, *banks, *own, mx.array([count], mx.int32))

        clean = run(bank)
        dirty = [mx.where((mx.arange(768) >= count)[None, None, :, None], 100, b)
                 .astype(q.dtype) for b in bank]
        assert same_bits(clean, run(dirty)).item()
        combined = [mx.concatenate([b[:, :, :count], v], axis=2)
                    for b, v in zip(bank, own, strict=True)]
        reference = mx.fast.scaled_dot_product_attention(q, *combined, scale=128 ** -0.5)
        assert mx.allclose(clean, reference, atol=0.003, rtol=0.02).item()


def test_replay_requalifies_at_attention_boundary_and_stops_after_mismatch():
    entry = InklingKVCache(1024)
    entry.draft = SimpleNamespace(entries=())
    model = SimpleNamespace(drafter=SimpleNamespace(blocks=()))
    engine = replay.RoundReplay(None, "round")
    entry.kv._offset = 4091
    assert engine.plan(model, [entry])
    engine.check([mx.array([1])], [mx.array([2])], entry.offset)
    assert engine.reason and not engine.plan(model, [entry])
    for offset in (4092, 4093, 4094):
        entry.kv._offset = offset
        assert not engine.plan(model, [entry])
    entry.kv._offset = 4095
    assert engine.plan(model, [entry])
    assert engine.program is None and engine.stage == "panel" and engine.checks == 0
