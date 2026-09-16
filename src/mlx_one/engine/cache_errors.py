"""Explicit failures raised by the cache runtime."""


class CacheError(RuntimeError):
    """Base class for cache-runtime failures."""


class CacheCapacityError(CacheError, MemoryError):
    """The configured cache budget cannot satisfy an allocation."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class CacheTopologyUnsupported(CacheError):
    """A requested backend cannot represent the model cache topology."""


class PrefixReuseUnsupported(CacheError):
    """The complete model state cannot be restored safely from a prefix."""


class KVQuantizationUnsupported(CacheError):
    """A requested KV quantization/backend combination is unsupported."""


class CacheReservationError(CacheError):
    """A cache reservation has an invalid lifecycle transition."""


class CacheOwnershipError(CacheError):
    """Cache ownership was released or transferred incorrectly."""


class CacheCorruptionError(CacheError):
    """Internal block or reference metadata is inconsistent."""
