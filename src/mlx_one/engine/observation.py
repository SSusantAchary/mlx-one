"""Private request-local qualification instrumentation; no native imports."""

from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar("mlx_one_request_observer", default=None)


def current_observer():
    return _observer.get()


@contextmanager
def observe_request(observer):
    token = _observer.set(observer)
    try:
        yield observer
    finally:
        _observer.reset(token)
