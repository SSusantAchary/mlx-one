"""Actor-local observations shared by adapters without HTTP dependencies."""
import threading

_slot_local = threading.local()


def current_slot():
    return int(getattr(_slot_local, "value", 0))


def current_queue_ms():
    return getattr(_slot_local, "queue_ms", None)


class QueueFullError(RuntimeError):
    pass


class RequestTimeoutError(TimeoutError):
    pass
