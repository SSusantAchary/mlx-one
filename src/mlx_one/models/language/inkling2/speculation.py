"""Greedy DSpark verification using the existing proposal lifecycle."""

import time

import mlx.core as mx

from mlx_one.engine.speculation import SpeculativeProposal
from mlx_one.ops.logits import LogitsMode

from .long_attention import _supported_gpu
from .verify import same_bits


def observe_draft(model, captures, cache):
    model.drafter.observe(captures, cache)
    mx.eval(*cache.tensors())


class InklingProposal(SpeculativeProposal):
    def propose_mtp(self, first, hidden, maximum):
        from mlx_one.engine.pressure import execution_forecast

        model = self.bundle.model
        draft = self.cache[0].draft
        count = min(maximum, model.drafter.width - 1)
        self.boundaries = [tuple(entry.boundary() for entry in self.cache)]
        self.captures = []
        self._pending_panel = None
        try:
            if self.runtime is not None:
                # Each potentially kept row can retain one version of the local window.
                windows = sum(c.allocated_nbytes for c in self.cache if c.max_size)
                amount = execution_forecast(
                    self.runtime, model.config, batch=1, chunk=count + 1,
                    history=self.cache[0].offset, phase="decode",
                    explicit_bytes=count * windows,
                )["bytes"]
                self.resources.enter_context(self.runtime.transient_reservation(amount))
            if self.cancelled():
                return self
            draft_started = time.perf_counter()
            guesses = model.drafter.propose(model, first, draft, count)
            if (getattr(model, "verify_panels", False) and mx.default_device() == mx.gpu
                    and _supported_gpu() and self._verify_panel(first, guesses, draft_started)):
                return self
            # The anchor must be verified regardless of the draft. Evaluate its
            # prediction together with the draft before reading either on the host.
            prediction = self._verify_one(first, pending=(guesses,), started=draft_started)
            guesses = guesses.tolist()
            self.drafted = len(guesses)
            self.candidates = [first, prediction]
            for guess in guesses:
                if self.cancelled() or guess != prediction:
                    break
                self.accepted += 1
                prediction = self._verify_one(guess)
                self.candidates.append(prediction)
            return self
        except BaseException:
            self.close()
            raise

    def _verify_panel(self, first, guesses, started):
        model = self.bundle.model
        rows = guesses.size + 1
        if not 2 <= rows <= 4 or self.cancelled():
            return False
        offset = self.cache[0].offset
        # Bucket below/above the attention dispatch boundary and by context tier.
        signature = (rows, offset.bit_length(), offset < 4095 <= offset + rows - 1)
        proof = model.verify_panel_proofs.get(signature)
        if proof is False:
            self.panel_declined = True
            return False

        guesses = guesses.astype(mx.int32)
        panel_ids = mx.concatenate([mx.array([first], mx.int32), guesses])
        logits, captures, tapes = model.verify_panel(panel_ids[None], self.cache)
        state = [t for c in self.cache for t in c.tensors()]
        if proof is None:
            mx.eval(panel_ids, logits, captures, state)
            self._prove_panel(signature, panel_ids.tolist(), logits, captures, tapes, offset)
            return True

        predictions = mx.argmax(logits[0], axis=-1).astype(mx.int32)
        matched = predictions[:-1] == guesses
        kept = mx.cumprod(matched.astype(mx.int32), axis=0).sum().astype(mx.int32)
        packet = mx.concatenate([kept[None], predictions])
        mx.eval(packet, captures, state)
        accepted, *predictions_host = packet.tolist()
        if getattr(model, "commit_panel_prefix", False):
            # Like MTPLX's round commit, select a single KV/conv prefix after
            # the caller applies stops and limits. Keep the old state for rollback.
            self._pending_panel = (offset, rows, captures, tapes)
        else:
            self.captures = [tuple(t[:, r:r + 1] for t in captures) for r in range(rows)]
            for kept in range(1, rows + 1):
                self.boundaries.append(tuple(tape.boundary(offset, kept, entry.max_size)
                                             for tape, entry in zip(tapes, self.cache, strict=True)))
        self.panel_used = True
        self.drafted = rows - 1
        self.accepted = accepted
        self.candidates = [first, *predictions_host[:accepted + 1]]
        self._observe_verify(started, rows, logits, "verify-panel")
        return True

    def _prove_panel(self, signature, ids, logits, captures, tapes, offset):
        # Copy only the newly written full-KV rows before replaying on these banks.
        panel_kv = [tuple(mx.asarray(x[:, :, offset:offset + len(ids)], copy=True)
                          for x in (c.kv.keys, c.kv.values)) if not c.max_size else None
                    for c in self.cache]
        mx.eval(logits, [t for pair in panel_kv if pair is not None for t in pair])
        self._restore(0)
        first_mismatch, predictions = None, []
        for index, token in enumerate(ids):
            if self.cancelled():
                return
            ref, prediction = self._forward_one(token)
            predictions.append(prediction)
            checks = []
            for visit, (entry, tape, global_kv) in enumerate(
                zip(self.cache, tapes, panel_kv, strict=True)
            ):
                _, window, conv = tape.boundary(offset, index + 1, entry.max_size)
                kv = (tuple(t[:, :, index:index + 1] for t in global_kv)
                      if window is None else tuple(t[:, :, -1:] for t in window))
                expected = (entry.kv.state() if entry.max_size else
                            (entry.kv.keys[:, :, offset + index:offset + index + 1],
                             entry.kv.values[:, :, offset + index:offset + index + 1]))
                checks.extend(
                    (f"visit {visit} KV {kind}", same_bits(a, b[:, :, -1:]))
                    for kind, a, b in zip(("key", "value"), kv, expected, strict=True)
                )
                checks.extend((f"visit {visit} convolution {slot}", same_bits(a, b))
                              for slot, (a, b) in enumerate(zip(conv, entry.conv, strict=True)))
            checks.extend((f"draft capture {tap}", same_bits(a[:, index:index + 1], b))
                          for tap, (a, b) in enumerate(zip(captures, ref.hidden_states, strict=True)))
            checks.append(("logits", same_bits(logits[:, index:index + 1], ref.logits)))
            verdicts = mx.stack([equal for _, equal in checks]).tolist()
            if first_mismatch is None:
                first_mismatch = next((f"row {index} {name}" for (name, _), equal
                                       in zip(checks, verdicts, strict=True) if not equal), None)
            self.captures.append(ref.hidden_states)
            self.boundaries.append(tuple(entry.boundary() for entry in self.cache))
        model = self.bundle.model
        proven = first_mismatch is None
        model.verify_panel_proofs[signature] = proven
        self.panel_declined = not proven
        if self.observer is not None:
            self.observer({"kind": "verification", "rows": len(ids), "offset": offset,
                           "passed": proven, "first_mismatch": first_mismatch,
                           "stages": {str(key): value
                                      for key, value in model.verify_stage_proofs.items()}})
        self._accept_panel(ids[0], ids[1:], predictions)

    def _accept_panel(self, first, guesses, predictions):
        self.drafted = len(guesses)
        self.candidates = [first, predictions[0]]
        for guess, expected, following in zip(guesses, predictions, predictions[1:]):
            if guess != expected or self.cancelled():
                break
            self.accepted += 1
            self.candidates.append(following)

    def _verify_one(self, token, *, pending=(), started=None):
        started = time.perf_counter() if started is None else started
        output, prediction = self._forward_one(token, pending)
        self.captures.append(output.hidden_states)
        self.boundaries.append(tuple(entry.boundary() for entry in self.cache))
        self._observe_verify(started, 1, output.logits, "verify")
        return prediction

    def _forward_one(self, token, pending=()):
        output = self.bundle.model(
            mx.array([[token]]), cache=self.cache, logits_mode=LogitsMode.LAST_ONLY,
            output_hidden_states=True,
        )
        prediction = mx.argmax(output.logits[0, -1])
        mx.eval(*pending, prediction, output.hidden_states,
                [t for c in self.cache for t in c.tensors()])
        return output, int(prediction.item())

    def _observe_verify(self, started, rows, logits, stage):
        if self.observer is not None:
            now = time.perf_counter()
            self.observer({
                "kind": "forward", "phase": "decode", "duration_ms": (now - started) * 1000,
                "at": now, "batch_size": 1, "input_tokens": rows,
                "speculative_stage": stage, "logits_dtype": str(logits.dtype),
            })

    def _restore(self, rows):
        for entry, state in zip(self.cache, self.boundaries[rows], strict=True):
            entry.restore(state)

    def commit(self, consumed):
        if self.cancelled():
            return
        if not 0 <= consumed <= len(self.candidates):
            raise ValueError("Inkling consumed tokens exceed verified candidates")
        rows = max(0, consumed - 1)
        captures = ()
        if self._pending_panel is not None:
            offset, width, panel_captures, tapes = self._pending_panel
            if rows == 0:
                self._restore(0)
            elif rows < width:
                for entry, tape in zip(self.cache, tapes, strict=True):
                    entry.restore(tape.boundary(offset, rows, entry.max_size))
            if rows:
                captures = tuple(t[:, :rows] for t in panel_captures)
        else:
            if rows < len(self.captures):
                self._restore(rows)
            if rows:
                captures = tuple(mx.concatenate(tap, axis=1)
                                 for tap in zip(*self.captures[:rows], strict=True))
        if captures:
            observe_draft(self.bundle.model, captures, self.cache[0].draft)
        self.boundaries.clear()
        self._pending_panel = None
        if self.handle is not None:
            self.handle.computed_tokens = self.cache[0].offset
            self.runtime.reconcile(self.handle)

    def close(self):
        if getattr(self, "boundaries", None):
            self._restore(0)
            self.boundaries.clear()
        self._pending_panel = None
        super().close()
