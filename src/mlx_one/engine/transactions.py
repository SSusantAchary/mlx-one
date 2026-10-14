"""Evaluate and validate complete request state before reversible ownership swaps."""
from __future__ import annotations


def component_tensors(component):
    """Inspect physical tensors; never materialize paged history."""
    if hasattr(component, '_blocks'):
        values = [ref.block.state for ref in component._blocks]
    else:
        values = [getattr(component, name, None) for name in
                  ('keys', 'values', '_keys', '_values', 'conv_state', 'recurrent_state')]
    def flatten(value):
        if isinstance(value, (tuple, list)):
            return [tensor for part in value for tensor in flatten(part)]
        return [value] if value is not None and hasattr(value, 'dtype') else []
    return tuple(tensor for value in values for tensor in flatten(value))


def release_all(bundles):
    error = None
    for bundle in bundles:
        try:
            bundle.release()
        except BaseException as exc:
            error = error or exc
    if error is not None:
        raise error


def commit_evaluated(requests, drafts):
    """Drafts receive old ownership after success; caller releases every draft.

    Component object identities stay stable for existing generation iterators.
    Swapping dictionaries transfers page/carry ownership without early releases.
    Failure restores all layer dictionaries and reservation counters.
    """
    import mlx.core as mx

    from mlx_one.engine.batching import request_cancelled

    if len(requests) != len(drafts) or len({r.request_id for r in requests}) != len(requests):
        raise ValueError('transaction request mapping mismatch')
    journal, accounts = [], []
    for request, draft in zip(requests, drafts, strict=True):
        first = requests[0]
        if (request.bundle is not first.bundle or request.model_epoch != first.model_epoch
                or request.phase != first.phase or not request.cache
                or len({c.offset for c in request.cache}) != 1):
            raise ValueError('transaction execution identity or boundary mismatch')
        if len(request.cache) != len(draft.entries):
            raise ValueError('transaction layer mapping mismatch')
        expected = request.cache[0].offset + len(request.token_ids)
        if any(c.offset != expected for c in draft.entries):
            raise ValueError('transaction committed position mismatch')
        if request.handle is not None:
            request.runtime._validate_handle(request.handle)
            if request.handle.bundle.entries is not request.cache:
                raise ValueError('transaction cache ownership mismatch')
        for original, replacement in zip(request.cache, draft.entries, strict=True):
            if type(original) is not type(replacement):
                raise ValueError('transaction component type mismatch')
            if not hasattr(original, '__dict__'):
                raise ValueError('component lacks transactional adoption contract')
        mx.eval(*(tensor for c in draft.entries for tensor in component_tensors(c)))
    # Recheck cancellation and epochs after evaluation, before any adoption.
    for request, draft in zip(requests, drafts, strict=True):
        if request_cancelled(request):
            continue
        if request.handle is not None:
            request.runtime._validate_handle(request.handle)
            handle = request.handle
            reservation = handle.reservation
            accounts.append((handle, handle.computed_tokens, handle.reserved_bytes,
                             reservation.committed_bytes if reservation is not None else None))
        journal.extend(zip(request.cache, draft.entries, strict=True))
    swapped = 0
    try:
        for original, replacement in journal:
            original.__dict__, replacement.__dict__ = replacement.__dict__, original.__dict__
            swapped += 1
        for request in requests:
            if request.handle is not None and not request_cancelled(request):
                request.handle.computed_tokens = request.cache[0].offset
                request.runtime.reconcile(request.handle)
    except BaseException:
        for original, replacement in reversed(journal[:swapped]):
            original.__dict__, replacement.__dict__ = replacement.__dict__, original.__dict__
        for handle, computed, reserved, committed in accounts:
            handle.computed_tokens, handle.reserved_bytes = computed, reserved
            if handle.reservation is not None:
                handle.reservation.committed_bytes = committed
        raise
