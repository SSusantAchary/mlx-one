"""Request-local flow-matching schedules for the pinned image checkpoints."""

from math import exp

import numpy as np


def flow_schedule(family: str, steps: int, image_tokens: int, *, z_shift: float = 3.0):
    """Return float32 sigma nodes including the terminal zero.

    Z-Image's API counts the terminal node. Its zero-to-zero interval does
    not update the sample, so the native pipeline evaluates only eight
    intervals at the default of nine nodes.
    """
    if family == "flux2-klein":
        early = 8.73809524e-05 * image_tokens + 1.89833333
        late = 0.00016927 * image_tokens + 0.45666666
        mu = late if image_tokens > 4300 else early + (late - early) * (steps - 10) / 190
        shift = np.float32(exp(mu))
        raw = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float32)
        sigmas = shift / (shift + (1.0 / raw - 1.0))
        return np.append(sigmas, np.float32(0))
    if family == "z-image-turbo":
        shift = np.float32(z_shift)
        raw = np.linspace(1.0, 0.0, steps, dtype=np.float32)
        return shift * raw / (1.0 + (shift - 1.0) * raw)
    raise ValueError(f"unknown diffusion schedule: {family}")
