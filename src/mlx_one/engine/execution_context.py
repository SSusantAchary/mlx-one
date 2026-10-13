"""Actor-local observations shared by adapters without HTTP dependencies."""
import threading
from contextvars import ContextVar

_slot_local = threading.local()


def current_slot():
    return int(getattr(_slot_local, "value", 0))


def current_queue_ms():
    return getattr(_slot_local, "queue_ms", None)


class QueueFullError(RuntimeError):
    pass


class RequestTimeoutError(TimeoutError):
    pass


# Installed only by qualification middleware; ordinary requests leave this empty.
qualification_request = ContextVar("qualification_request", default=None)
