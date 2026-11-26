import json
from contextlib import nullcontext
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_one.core.precision import validate_finite_safetensors
from mlx_one.models.language.inkling2.attention import stock_attention
from mlx_one.models.language.inkling2.attention_metal import grouped_sink_attention
from mlx_one.models.language.inkling2.cache import InklingKVCache
from mlx_one.models.language.inkling2.speculation import InklingProposal


@pytest.mark.parametrize("payload", [b"\x7e\xfe", b"\x7f", b"\xff"])
def test_e4m3_finite_scan(tmp_path, payload):
    header = json.dumps({
        "scales": {"dtype": "F8_E4M3", "shape": [len(payload)],
                   "data_offsets": [0, len(payload)]},
    }).encode()
    path = tmp_path / "weights.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    if len(payload) == 2:
        validate_finite_safetensors(path)
    else:
        with pytest.raises(ValueError, match="non-finite checkpoint tensor: scales"):
            validate_finite_safetensors(path)


@pytest.mark.parametrize(("window", "expected"), [
    (0, [[3.0, 6.5], [6.0, 13.0]]),
    (2, [[5.0, 12.0], [10.0, 24.0]]),
])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal attention required")
def test_grouped_sinks_keep_chronological_windows(window, expected):
    q = mx.zeros((1, 4, 2, 128), mx.bfloat16)
    k = mx.zeros((1, 2, 5, 128), mx.bfloat16)
    rows = mx.array([[0, 0, 3, 12, 24], [0, 0, 6, 24, 48]], dtype=mx.bfloat16)
    v = mx.broadcast_to(rows[None, :, :, None], k.shape)
    profiles = mx.zeros((1, 2, 4, 2), mx.bfloat16)
    sinks = mx.zeros((1, 2, 4), mx.bfloat16)
    candidate = grouped_sink_attention(q, k, v, profiles, sinks, window)
    reference = stock_attention(q, k, v, profiles, sinks, window)
    assert mx.array_equal(candidate, reference).item()
    assert candidate[0, ::2, :, 0].tolist() == expected


@pytest.mark.parametrize("cancel", [False, True])
def test_draft_verification_keeps_only_consumed_cache_rows(cancel):
    class Draft:
        width = 8
        observed = []

        def propose(self, model, anchor, state, count):
            assert anchor == 1
            return mx.array([2, 9])

        def observe(self, captures, state):
            self.observed.extend(captures[0].tolist())

    class Target:
        drafter = Draft()

        def __call__(self, token, *, cache, logits_mode, output_hidden_states):
            assert token.shape == (1, 1)
            for entry in cache:
                rows = token.reshape(1, 1, 1, 1)
                entry.kv.update(rows, rows)
                entry.conv[0] = token
            logits = (mx.arange(10) == token[0, 0] + 1).astype(mx.float32)
            return SimpleNamespace(logits=logits[None, None], hidden_states=(token,))

    entries = (InklingKVCache(), InklingKVCache(2))
    for entry in entries:
        rows = mx.array([7, 8]).reshape(1, 1, 2, 1)
        entry.kv.update(rows, rows)
        entry.conv[0] = mx.array([[8]])
        mx.eval(*entry.tensors())
    entries[0].draft = SimpleNamespace(tensors=lambda: ())
    stopped = False
    model = Target()
    proposal = InklingProposal(SimpleNamespace(model=model), entries, cancelled=lambda: stopped)
    proposal.propose_mtp(1, None, 2)
    assert proposal.candidates == [1, 2, 3]
    assert (proposal.drafted, proposal.accepted) == (2, 1)
    stopped = cancel
    proposal.commit(2)
    proposal.close()
    expected = [7, 8] if cancel else [7, 8, 1]
    assert [c.offset for c in entries] == [len(expected)] * 2
    assert entries[0].kv.state()[0].flatten().tolist() == expected
    assert entries[1].kv.state()[0].flatten().tolist() == expected[-2:]
    assert entries[1].conv[0].item() == expected[-1]
    assert model.drafter.observed == ([] if cancel else [[1]])


@pytest.mark.parametrize(("consumed", "cancel"), [(0, False), (1, False), (3, False),
                                                 (5, False), (5, True)])
@pytest.mark.parametrize("async_commit", [False, True])
def test_qualified_panel_commits_only_the_consumed_prefix(
    monkeypatch, consumed, cancel, async_commit,
):
    from mlx_one.models.language.inkling2 import speculation
    from mlx_one.models.language.inkling2.verify import VisitTape

    monkeypatch.setattr(speculation, "_supported_gpu", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    async_calls = []
    monkeypatch.setattr(mx, "async_eval", lambda *tensors: async_calls.append(tensors))
    observed = []

    class Target:
        verify_panels = commit_panel_prefix = True
        async_draft_commit = async_commit
        verify_panel_proofs = {(4, 2, False): True}
        drafter = SimpleNamespace(
            width=8, propose=lambda *args: mx.array([2, 3, 4]),
            observe=lambda captures, _: observed.extend(captures[0][0].tolist()),
        )

        def verify_panel(self, ids, cache):
            tapes = [VisitTape(ids.shape[1]) for _ in cache]
            for entry, tape in zip(cache, tapes, strict=True):
                kv = ids.reshape(1, 1, -1, 1)
                state = entry.kv.update(kv, kv)
                if entry.max_size:
                    tape.window = state
                for slot, tail in enumerate(entry.conv):
                    tape.record_conv(slot, tail, ids[..., None], 2)
                    entry.conv[slot] = tape.conv_histories[slot][:, -2:]
            predictions = ids[..., None] + 1
            return (predictions == mx.arange(8)).astype(mx.float32), (ids,), tapes

    entries = (InklingKVCache(), InklingKVCache(2))
    for entry in entries:
        history = mx.array([7, 8]).reshape(1, 1, 2, 1)
        entry.kv.update(history, history)
        entry.conv = [history.reshape(1, 2, 1).astype(mx.float32)] * 4
        mx.eval(*entry.tensors())
    entries[0].draft = SimpleNamespace(tensors=lambda: ())
    cancelled = False
    proposal = InklingProposal(
        SimpleNamespace(model=Target()), entries, cancelled=lambda: cancelled,
    )
    proposal.propose_mtp(1, None, 3)
    assert proposal.panel_used and proposal.candidates == [1, 2, 3, 4, 5]
    assert len(proposal.boundaries) == 1
    cancelled = cancel
    proposal.commit(consumed)
    proposal.close()

    kept = list(range(1, consumed)) if not cancel else []
    expected = [7, 8, *kept]
    assert observed == kept
    assert len(async_calls) == int(async_commit and bool(kept))
    for entry in entries:
        assert entry.offset == len(expected)
        for bank in entry.kv.state():
            assert bank.flatten().tolist() == (expected[-2:] if entry.max_size else expected)
        assert all(tail.flatten().tolist() == expected[-2:] for tail in entry.conv)


def test_http_admission_uses_native_loop_and_draft_reservation(monkeypatch):
    from mlx_one.engine.memory import AdmissionStatus, MemoryBudget
    from mlx_one.models.language.inkling2.model import InklingForCausalLM
    from mlx_one.server import admission

    model = InklingForCausalLM.__new__(InklingForCausalLM)
    model.config = SimpleNamespace(max_position_embeddings=131072)
    model.cfg = {"num_key_value_heads": 8, "head_dim": 128, "sconv_kernel_size": 4,
                 "hidden_size": 2304, "sliding_window_size": 1024}
    model.config.draft = {"num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2304}
    model.layers = [SimpleNamespace(sliding=False), SimpleNamespace(sliding=True)]
    model.visit_order = (0, 1, 1)
    model.drafter = SimpleNamespace(blocks=[SimpleNamespace(window=None)])
    cache = model.make_cache()

    gib = 1024**3
    plan = admission.MemoryAdmission(MemoryBudget.from_total(32 * gib, process_bytes=23 * gib))
    monkeypatch.setattr(admission, "get_memory_snapshot",
                        lambda: SimpleNamespace(active_bytes=15 * gib))
    bundle = SimpleNamespace(model=model)
    plain = plan.decide(bundle, 131072)
    draft = plan.decide(bundle, 131072, speculative=True)
    assert plain.requested_bytes == model.configure_request_cache(cache, 131072)
    assert plain.status is draft.status is AdmissionStatus.ADMIT
    assert draft.requested_bytes - plain.requested_bytes == 4096 * 131072 + 4608
    assert cache[0].kv.step == 131072
    assert all(c.kv.keys is None for c in cache)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal convolution required")
@pytest.mark.parametrize(("width", "dynamic", "residual"), [
    (1024, False, None), (1024, True, None), (2304, False, "float32"),
    (2304, False, "bfloat16"),
])
def test_fused_one_row_short_convolution_matches_reference(width, dynamic, residual):
    from mlx_one.models.language.inkling2 import short_conv

    keys = mx.random.split(mx.random.key(width + 47), 5)
    x = mx.random.normal((1, 1, width), key=keys[0]).astype(mx.bfloat16)
    prev = mx.random.normal((1, 3, width), key=keys[1])
    weight = mx.random.normal((width, 4), key=keys[2]).T
    change = (
        mx.random.normal((1, 1, 32), key=keys[3]).astype(mx.bfloat16) if dynamic else None
    )
    add = mx.random.normal(x.shape, key=keys[4]) if residual is not None else None
    dtype = getattr(mx, residual) if residual is not None else None
    args = (x, prev, weight, change, add, dtype)
    expected = short_conv.reference(*args)
    actual = short_conv.fused(*args)
    if actual is None:
        pytest.skip("This GPU did not pass the first-use bitwise check")
    for a, b in zip(actual, expected, strict=True):
        word = mx.uint32 if a.dtype == mx.float32 else mx.uint16
        assert mx.array_equal(a.view(word), b.view(word)).item()


def test_short_convolution_falls_back_for_prompt_and_before_tail():
    from mlx_one.models.language.inkling2 import short_conv

    x = mx.zeros((1, 2, 1024), mx.bfloat16)
    weight = mx.zeros((4, 1024))
    assert short_conv.fused(x, None, weight, None, None, None) is None
    assert short_conv.fused(x, mx.zeros((1, 3, 1024)), weight, None, None, None) is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal convolution required")
@pytest.mark.parametrize(("dynamic", "cold"), [(False, True), (False, False), (True, False)])
def test_fused_prefill_convolution_matches_chunk_reference(dynamic, cold):
    from mlx_one.models.language.inkling2 import short_conv
    from mlx_one.models.language.inkling2.verify import same_bits

    keys = mx.random.split(mx.random.key(128), 5)
    x = mx.random.normal((1, 128, 1024), key=keys[0]).astype(mx.bfloat16)
    prev = None if cold else mx.random.normal((1, 3, 1024), key=keys[1])
    weight = mx.random.normal((1024, 4), key=keys[2]).T
    change = mx.random.normal((1, 128, 32), key=keys[3]).astype(x.dtype) if dynamic else None
    add = mx.random.normal(x.shape, key=keys[4])
    args = (x, prev, weight, change, add, mx.float32)
    proofs = {}
    expected = short_conv.reference(*args)
    first = short_conv.fused_prefill(*args, proofs=proofs)
    assert proofs
    assert all(same_bits(a, b).item() for a, b in zip(first, expected, strict=True))
    repeat = short_conv.fused_prefill(*args, proofs=proofs)
    if all(proofs.values()):
        assert all(same_bits(a, b).item() for a, b in zip(repeat, expected, strict=True))
    else:
        assert repeat is None


def test_unproven_prefill_convolution_uses_reference(monkeypatch):
    from mlx_one.models.language.inkling2 import short_conv
    from mlx_one.models.language.inkling2.verify import same_bits

    x = mx.zeros((1, 128, 1024), mx.bfloat16)
    weight = mx.zeros((4, 1024))
    monkeypatch.setattr(short_conv, "_supported", lambda *args: True)
    monkeypatch.setattr(short_conv, "_launch", lambda *args: (mx.ones_like(x), args[1]))
    proofs = {}
    args = (x, None, weight, None, None, None)
    out, tail = short_conv.fused_prefill(*args, proofs=proofs)
    assert proofs and not any(proofs.values())
    assert same_bits(out, x).item() and tail.shape == (1, 3, 1024)
    assert short_conv.fused_prefill(*args, proofs=proofs) is None
    assert short_conv.fused_prefill(x[:, :1], *args[1:], proofs={}) is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal scatter required")
def test_expert_scatter_matches_inverse_sort_on_strided_rows(monkeypatch):
    from mlx_one.models.language.inkling2 import expert_scatter

    proofs = {}
    monkeypatch.setattr(expert_scatter, "_qualified", proofs)
    routes = mx.arange(1024, dtype=mx.uint32) % 53
    order = mx.argsort(routes)
    base = mx.arange(1024 * 32, dtype=mx.uint16).reshape(1024, 32)
    rows = base.view(mx.bfloat16)[:, ::2]
    reference = rows[mx.argsort(order)]
    result = expert_scatter.restore(rows, order)
    assert proofs == {rows.shape: True}
    assert mx.array_equal(result.view(mx.uint16), reference.view(mx.uint16)).item()
    assert expert_scatter.restore(rows[:, :-1], order) is None


@pytest.mark.parametrize("window, capacity", [(None, 8), (None, 2), (2, 8)])
def test_dspark_proposal_kv_preserves_committed_history(window, capacity):
    from mlx_one.core.cache import KVCache
    from mlx_one.core.sliding_cache import SlidingKVCache
    from mlx_one.models.language.inkling2.draft import DraftBlock

    entry = SlidingKVCache(window) if window else KVCache()
    entry.step = capacity
    prefix = mx.array([1, 2], dtype=mx.bfloat16).reshape(1, 1, 2, 1)
    candidate = mx.array([3, 4], dtype=mx.bfloat16).reshape(1, 1, 2, 1)
    entry.update(prefix, prefix * 2)
    block = SimpleNamespace(window=window)

    key, value = DraftBlock.proposal_kv(block, entry, candidate, candidate * 2)
    mx.eval(key, value)
    assert key.flatten().tolist() == [1, 2, 3, 4]
    assert value.flatten().tolist() == [2, 4, 6, 8]
    assert entry.offset == 2
    live_key, live_value = entry.state()
    assert mx.array_equal(live_key, prefix).item()
    assert mx.array_equal(live_value, prefix * 2).item()
    committed = mx.array([7], dtype=mx.bfloat16).reshape(1, 1, 1, 1)
    entry.update(committed, committed * 2)
    assert entry.state()[0].flatten().tolist() == ([2, 7] if window else [1, 2, 7])


@pytest.mark.parametrize(("rows", "count"), [(17, 20225), (1, 65537)])
def test_long_global_attention_matches_native_bits_with_reserved_tail(rows, count, monkeypatch):
    from mlx_one.models.language.inkling2 import long_attention
    from mlx_one.models.language.inkling2.attention import attend

    if not long_attention._supported_gpu():
        pytest.skip("This exact reduction plan requires the base M4 and MLX 0.32.2")
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    proofs = {}
    monkeypatch.setattr(long_attention, "_qualified", proofs)
    def bf(shape, seed):
        return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)

    q = bf((1, rows, 24, 128), 1).transpose(0, 2, 1, 3)
    k, v = (bf((1, 8, count + 17, 128), seed) for seed in (2, 3))
    k[:, :, count:] = float("nan")
    v[:, :, count:] = float("nan")
    profiles, sink = bf((1, rows, 24, 1024), 4), bf((1, rows, 24), 5)
    args = (q, k, v, profiles, sink, count)
    reference = attend(q, k[:, :, :count], v[:, :, :count], profiles, sink, 0)
    actual = long_attention.try_global_attention(*args)
    assert proofs and all(proofs.values())  # a silent fallback must fail this test
    assert mx.array_equal(actual.view(mx.uint16), reference.view(mx.uint16)).item()
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "128")
    assert long_attention.try_global_attention(*args) is None


def test_long_global_attention_preserves_reference_on_failed_proof(monkeypatch):
    from mlx_one.models.language.inkling2 import long_attention

    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    monkeypatch.setattr(long_attention, "_supported_gpu", lambda: True)
    monkeypatch.setattr(long_attention, "_qualified", {})
    query = mx.zeros((1, 24, 1, 128), mx.bfloat16)
    keys = mx.broadcast_to(mx.array(0, mx.bfloat16), (1, 8, 20225, 128))
    profiles = mx.zeros((1, 1, 24, 1024), mx.bfloat16)
    sink = mx.zeros((1, 1, 24), mx.bfloat16)
    monkeypatch.setattr(long_attention, "_launch", lambda *args: mx.ones_like(query))
    monkeypatch.setattr(long_attention, "attend", lambda *args: query)
    args = (query, keys, keys, profiles, sink, 20225)
    assert long_attention.try_global_attention(*args) is query
    assert long_attention.try_global_attention(*args) is None
    assert long_attention._plan(128, 1024) == ()
    assert long_attention._plan(2, 65536) == ()


@pytest.mark.parametrize(("window", "rows", "count"), [
    (0, 63, 1025),
    (0, 128, 8192),  # two query rows use the stock vector reduction
    (1024, 73, 1097),  # partially filled tiles at each window edge
    (1024, 128, 1152),
])
def test_flash_prompt_matches_stock_tiles(window, rows, count, monkeypatch):
    from mlx_one.models.language.inkling2 import prompt_attention

    if not prompt_attention.available():
        pytest.skip("This prefill kernel requires the base M4 and MLX 0.32.2")
    proofs = {}
    monkeypatch.setattr(prompt_attention, "_ready", proofs)

    def bf(shape, seed):
        return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)

    query = bf((1, rows, 24, 128), 1).transpose(0, 2, 1, 3)
    key, value = (bf((1, 8, count + 17, 128), seed) for seed in (2, 3))
    key[:, :, count:] = float("nan")
    value[:, :, count:] = float("nan")
    profiles, sink = bf((1, rows, 24, 1024), 4), bf((1, rows, 24), 5)
    actual = prompt_attention.prompt_attention(
        query, key, value, profiles, sink, window, count=count,
    )
    reference = stock_attention(
        query, key[:, :, :count], value[:, :, :count], profiles, sink, window,
    )
    assert proofs == {window: True}
    assert mx.array_equal(actual.view(mx.uint16), reference.view(mx.uint16)).item()


def test_adaptive_verification_uses_committed_throughput_and_falls_back():
    from mlx_one.models.language.inkling2.speculation import AdaptiveVerification, panel_signature

    policy = AdaptiveVerification()
    offset = 8192
    for _ in range(3):
        policy.record_b1(0.1)
        assert policy.choose(3, offset, {}) == 0
    policy.record_b1(0.1)
    assert policy.choose(3, offset, {}) == 1
    policy.record_panel(1, 0.22, 2)
    assert policy.choose(3, offset, {}) == 3
    policy.record_panel(3, 0.31, 4)
    assert policy.choose(3, offset, {}) == 3
    assert policy.choose(1, offset, {}) == 0
    assert policy.choose(3, offset, {panel_signature(4, offset): False}) == 0
    for _ in range(12):
        policy.record_panel(3, 0.31, 2)
    assert policy.choose(3, offset, {}) == 0
    while policy.rounds < 32:
        policy.record_b1(0.1)
    assert policy.choose(3, offset, {}) == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal projections required")
def test_attention_projection_bank_matches_individual_affine_products():
    from mlx_one.models.language.inkling2.model import AttentionProjections
    from mlx_one.models.language.inkling2.precision import AffineWeight

    weights = []
    for size in (24, 16, 32):
        matrix = mx.random.normal((size, 128), key=mx.random.key(size)).astype(mx.bfloat16)
        weights.append(AffineWeight(*mx.quantize(matrix, group_size=64, bits=4), 4, 64))
    group = AttentionProjections.prepare(weights)
    x = mx.random.normal((1, 1, 128), key=mx.random.key(2)).astype(mx.bfloat16)
    assert group is not None
    expected = [w.product(x) for w in weights]
    for actual, reference in zip(group(x), expected, strict=True):
        assert mx.array_equal(actual.view(mx.uint16), reference.view(mx.uint16)).item()
    assert group.proofs.keys() == {(1, False, mx.bfloat16)}


@pytest.mark.parametrize(
    ("limit", "eos_id", "matches", "want", "rounds"),
    [
        (1, None, True, [9], 0),
        (7, None, True, list(range(9, 16)), 2),
        (5, None, False, list(range(9, 14)), 3),
        (7, 10, True, [9], 1),
        (7, 11, True, [9, 10], 1),
    ],
)
def test_continuous_dspark_streams_successors_and_retains_only_verified_prefix(
    monkeypatch, limit, eos_id, matches, want, rounds,
):
    from mlx_one.models.language.inkling2 import long_attention, speculation
    from mlx_one.models.language.inkling2.verify import VisitTape
    from mlx_one.text.generation import _stream_bundle
    from mlx_one.text.schemas import TextGenerationOptions

    monkeypatch.setattr(long_attention, "_supported_gpu", lambda: True)
    monkeypatch.setattr(speculation, "_supported_gpu", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)

    class Target:
        continuous_dspark = verify_panels = True
        config = SimpleNamespace(hidden_size=8, num_attention_heads=1)
        verify_panel_proofs = {
            (rows, tier, False): True for rows in (2, 3, 4) for tier in (2, 3, 4)
        }

        def __init__(self):
            self.drafter = SimpleNamespace(width=8, propose=self.draft, observe=self.observe)
            self.ordinary_inputs = []

        def draft(self, model, anchor, state, count):
            start = anchor + 1 if matches else 25
            return mx.arange(start, start + count, dtype=mx.int32)

        @staticmethod
        def observe(captures, state):
            state.committed.extend(captures[0][0].tolist())

        @staticmethod
        def prefill_width(maximum, prompt_tokens):
            return maximum

        @staticmethod
        def forward(ids, cache, tapes=None):
            for visit, entry in enumerate(cache):
                kv = ids.reshape(1, 1, -1, 1)
                window = entry.kv.update(kv, kv)
                for slot, tail in enumerate(entry.conv):
                    history = ids[..., None].astype(mx.float32)
                    if tail is not None:
                        history = mx.concatenate((tail, history), axis=1)
                    entry.conv[slot] = history[:, -2:]
                    if tapes is not None:
                        tapes[visit].record_conv(slot, tail, ids[..., None], 2)
                if tapes is not None and entry.max_size:
                    tapes[visit].window = window
            return (ids + 1)[..., None] == mx.arange(32)

        def __call__(self, ids, *, cache, logits_mode, output_hidden_states):
            self.ordinary_inputs.append(ids[0].tolist())
            logits = self.forward(ids, cache).astype(mx.float32)
            return SimpleNamespace(logits=logits[:, -1:], last_hidden_state=ids,
                                   hidden_states=(ids,))

        def verify_panel(self, ids, cache):
            tapes = [VisitTape(ids.shape[1]) for _ in cache]
            return self.forward(ids, cache, tapes).astype(mx.float32), (ids,), tapes

    model = Target()
    cache = (InklingKVCache(), InklingKVCache(2))
    cache[0].draft = SimpleNamespace(committed=[], tensors=lambda: ())
    tokenizer = SimpleNamespace(encode=lambda _: [], bos_token_id=None, eos_token_id=eos_id)
    bundle = SimpleNamespace(architecture="inkling2_loop", model=model, tokenizer=tokenizer,
                             eos_token_ids=(), context_length=32)
    runtime = SimpleNamespace(transient_reservation=lambda _: nullcontext())
    stats = {}
    chunks = list(_stream_bundle(
        bundle, "", TextGenerationOptions(max_tokens=limit, temperature=0), None,
        cache=cache, _cache_runtime=runtime, _prepared_input_ids=(7, 8), spec_type="draft-mtp",
        speculative_stats=stats, _token_events_only=True,
    ))
    assert [chunk.token_id for chunk in chunks if chunk.token_id is not None] == want
    assert chunks[-1].terminal_token_id == eos_id
    assert stats.get("continuous_rounds", 0) == rounds
    assert model.ordinary_inputs == ([[7, 8], [12]] if rounds == 3 else [[7, 8]])
    committed = [7, 8, *want[:-1]]
    assert cache[0].draft.committed == committed
    for entry in cache:
        expected = committed[-2:] if entry.max_size else committed
        assert entry.kv.state()[0].flatten().tolist() == expected
        assert all(tail.flatten().tolist() == committed[-2:] for tail in entry.conv)


def test_prefill_overlapping_tiles_keep_the_same_bits(monkeypatch):
    from mlx_one.models.language.inkling2 import attention

    monkeypatch.setattr(attention, "query_tile_rows", lambda *args: 4)

    def bf(shape, seed):
        return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)

    query = bf((1, 4, 11, 128), 1)
    key, value = (bf((1, 2, 17, 128), n) for n in (2, 3))
    profiles, sinks = bf((1, 11, 4, 16), 4), bf((1, 11, 4), 5)
    args = (query, key, value, profiles, sinks, 0)
    serial = attention.stock_attention(*args)
    overlap = attention.stock_attention(*args, overlap_tiles=True)
    assert mx.array_equal(serial.view(mx.uint16), overlap.view(mx.uint16)).item()


@pytest.mark.parametrize(("rows", "count"), [(2, 819), (4, 1026), (4, 1028)])
def test_window_panel_attention_matches_chronological_bits(rows, count):
    from mlx_one.models.language.inkling2.attention import chronological_attention
    from mlx_one.models.language.inkling2.attention_metal import window_panel_attention
    from mlx_one.models.language.inkling2.long_attention import _supported_gpu

    if not _supported_gpu():
        pytest.skip("The window panel requires the base M4 and MLX 0.32.2")

    def bf(shape, seed):
        return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)

    q = bf((1, rows, 24, 128), 1).transpose(0, 2, 1, 3)
    k, v = (bf((1, count, 8, 128), n).transpose(0, 2, 1, 3) for n in (2, 3))
    p, s = bf((1, rows, 24, 1024), 4), bf((1, rows, 24), 5)
    args = (q, k, v, p, s, 1024)
    expected = chronological_attention(*args)
    candidate = window_panel_attention(*args)
    assert mx.array_equal(expected.view(mx.uint16), candidate.view(mx.uint16)).item()
