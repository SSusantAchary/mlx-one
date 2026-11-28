"""Native image pipeline and its device-owned component lifecycle."""

from __future__ import annotations

import io
import threading
import time
from concurrent.futures import CancelledError, Future
from pathlib import Path

from mlx_one.engine.device_executor import DeviceExecutor
from mlx_one.engine.execution_context import QueueFullError
from mlx_one.engine.memory import AdmissionStatus, MemoryBudget, MemoryUsage, UnifiedMemoryPlanner
from mlx_one.engine.pressure import sample_telemetry

from .checkpoint import load_component, resolve_checkpoint
from .profiles import image_profile, published_q4_lock
from .request import ImageRequest, ImageResult
from .schedule import flow_schedule

_GIB = 1024**3


class ImageMemoryError(RuntimeError):
    """The diffusion working set exceeds current unified-memory headroom."""


def _require_active(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise CancelledError("image generation cancelled")


class ImageEngine:
    """Generate one PNG with a pinned native FLUX.2 Klein or Z-Image pipeline."""

    def __init__(
        self,
        model: str,
        *,
        quantization: int | None = None,
        checkpoint: str | Path | None = None,
        offline: bool = False,
        idle_ttl: float = 120,
        queue_size: int = 2,
        process_ceiling_bytes: int | None = None,
    ) -> None:
        self.profile = image_profile(model)
        self.model = model
        self._published = model == self.profile.q4_model
        if quantization is not None and (
            type(quantization) is not int or quantization not in (4, 8)
        ):
            raise ValueError("image quantization must be 4, 8, or omitted for BF16")
        if self._published:
            if quantization not in (None, 4):
                raise ValueError("the published checkpoint already uses Q4")
            quantization = 4
        if not 0 <= idle_ttl < float("inf") or type(queue_size) is not int or queue_size < 1:
            raise ValueError("idle_ttl must be nonnegative and queue_size must be positive")
        if process_ceiling_bytes is not None and process_ceiling_bytes <= 0:
            raise ValueError("process_ceiling_bytes must be positive")
        self.quantization = quantization
        self.checkpoint = checkpoint
        self.offline = offline
        self.idle_ttl = idle_ttl
        self._queue_size = queue_size
        self._process_ceiling = process_ceiling_bytes
        self._lock = threading.RLock()
        self._device = DeviceExecutor.shared()
        self._key = ("image", id(self))
        self._closed = False
        self._pending: dict[Future, threading.Event] = {}
        self._timer: threading.Timer | None = None
        self._last_use = 0.0
        self._components: dict = {}
        self._checkpoint = None

    def __enter__(self) -> ImageEngine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def submit(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        steps: int | None = None,
        seed: int | None = None,
        cancel: threading.Event | None = None,
    ) -> Future[ImageResult]:
        """Enqueue a bounded request with cooperative cancellation."""
        request = ImageRequest.create(self.profile, prompt, width, height, steps, seed)
        cancel = cancel or threading.Event()
        with self._lock:
            if self._closed:
                raise RuntimeError("image engine is closed")
            if len(self._pending) >= self._queue_size + 1:
                raise QueueFullError("image generation queue is full")
            future = self._device.submit(
                self._key, lambda: self._generate(request, cancel), cancel=cancel
            )
            self._pending[future] = cancel
            future.add_done_callback(self._finished)
            return future

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        steps: int | None = None,
        seed: int | None = None,
    ) -> ImageResult:
        """Generate a single PNG and return it with execution metadata."""
        if self._device.is_owner:
            raise RuntimeError("synchronous generation cannot run on the device executor")
        cancel = threading.Event()
        future = self.submit(
            prompt, width=width, height=height, steps=steps, seed=seed, cancel=cancel
        )
        try:
            return future.result()
        except KeyboardInterrupt:
            cancel.set()
            future.cancel()
            raise

    def unload(self) -> None:
        """Wait for preceding requests to finish and unload resident weights."""
        self._device.call(self._key, self._release_weights)

    def close(self) -> None:
        """Cancel queued work and release weights after active GPU work completes."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for cancel in self._pending.values():
                cancel.set()
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self.unload()

    def _finished(self, future: Future) -> None:
        with self._lock:
            self._pending.pop(future, None)
            if self._pending or self._closed:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._last_use = time.monotonic()
            self._timer = threading.Timer(self.idle_ttl, self._expire, args=(self._last_use,))
            self._timer.daemon = True
            self._timer.start()

    def _expire(self, last_use: float) -> None:
        def if_idle() -> None:
            with self._lock:
                if not self._pending and self._last_use == last_use:
                    self._release_weights()
                    self._timer = None

        try:
            self._device.call(self._key, if_idle)
        except QueueFullError:
            with self._lock:
                if not self._closed and not self._pending and self._last_use == last_use:
                    self._timer = threading.Timer(1, self._expire, args=(last_use,))
                    self._timer.daemon = True
                    self._timer.start()

    def _release_weights(self) -> None:
        import sys

        mx = sys.modules.get("mlx.core")
        if self._components and mx is not None:
            mx.synchronize()
            self._components.clear()
            mx.clear_cache()

    def _admit(self, mx, request: ImageRequest):
        telemetry = sample_telemetry()
        installed = telemetry.installed_bytes or mx.device_info().get("memory_size")
        if not installed or telemetry.os_pressure in {"warning", "critical"}:
            raise ImageMemoryError("image generation requires verified unified-memory headroom")
        ceiling = min(
            value
            for value in (telemetry.recommended_bytes, int(installed * 0.75), self._process_ceiling)
            if value is not None
        )
        planner = UnifiedMemoryPlanner(MemoryBudget.from_total(installed, process_bytes=ceiling))
        if self._published:
            weights = {
                component: sum(item["size"] for item in shards.values()) / _GIB
                for component, shards in published_q4_lock(self.profile)["weights"].items()
            }
        else:
            flux = self.profile.family == "flux2-klein"
            denoiser_weights = {None: 8.0, 4: 2.5, 8: 4.5} if flux else {None: 12.5, 4: 4.0, 8: 7.0}
            weights = {
                "text_encoder": 6.5 if flux else 8.0,
                "transformer": denoiser_weights[self.quantization],
                "vae": 0.4,
            }
        cold = sum(weights.values()) if self.quantization else max(weights.values())
        resident = sum(weights[name] for name in self._components)
        scratch = 1.5 + request.width * request.height / 512**2 * 0.65
        estimate = int((cold - resident + scratch) * _GIB)
        active = (telemetry.mlx_active_bytes or 0) + (telemetry.mlx_cached_bytes or 0)
        usage = MemoryUsage(active_mlx_bytes=max(active, telemetry.rss_bytes or 0))
        decision = planner.admit(estimate, usage, queue_when_busy=False)
        if decision.status != AdmissionStatus.ADMIT and telemetry.mlx_cached_bytes:
            mx.synchronize()
            mx.clear_cache()
            sampled = sample_telemetry()
            usage = MemoryUsage(
                active_mlx_bytes=max(sampled.mlx_active_bytes or 0, sampled.rss_bytes or 0)
            )
            decision = planner.admit(estimate, usage, queue_when_busy=False)
        if decision.status != AdmissionStatus.ADMIT:
            raise ImageMemoryError(
                f"image generation needs {estimate / _GIB:.1f} GiB of additional headroom, "
                f"{decision.available_bytes / _GIB:.1f} GiB available"
            )
        return telemetry

    def _component(self, name: str):
        if name in self._components:
            return self._components[name]
        from mlx_one.models.image_generation.conditioning import ImageConditioner
        from mlx_one.models.image_generation.flux2 import Flux2Denoiser
        from mlx_one.models.image_generation.vae import ImageDecoder
        from mlx_one.models.image_generation.zimage import ZImageDenoiser

        checkpoint = self._checkpoint
        config = checkpoint.config(name)
        if name == "text_encoder":
            module = ImageConditioner(
                config,
                checkpoint.root / "tokenizer",
                self.profile.family,
                self.profile.max_prompt_tokens,
            )
        elif name == "transformer":
            factory = Flux2Denoiser if self.profile.family == "flux2-klein" else ZImageDenoiser
            module = factory(config)
        else:
            module = ImageDecoder(config)
        result = load_component(checkpoint, name, module, self.quantization)
        self._components[name] = result
        return result

    def _phase_done(self, mx, name: str) -> None:
        if self.quantization is None:
            del self._components[name]
            mx.synchronize()
            mx.clear_cache()

    def _generate(self, request: ImageRequest, cancel: threading.Event) -> ImageResult:
        import mlx.core as mx
        import numpy as np

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("image generation requires mlx-one[vision]") from exc
        start = time.perf_counter()
        timings = {"load": 0.0}
        _require_active(cancel)
        before = self._admit(mx, request)
        stage = time.perf_counter()
        if self._checkpoint is None:
            self._checkpoint = resolve_checkpoint(
                self.profile,
                self.checkpoint,
                self.quantization,
                offline=self.offline,
                published=self._published,
            )
        timings["load"] += time.perf_counter() - stage

        def component(name):
            _require_active(cancel)
            since = time.perf_counter()
            module = self._component(name)
            timings["load"] += time.perf_counter() - since
            return module

        try:
            encoder = component("text_encoder")
            stage = time.perf_counter()
            prompt = encoder(request.prompt)
            mx.eval(prompt)
            timings["text_encoder"] = time.perf_counter() - stage
            del encoder
            self._phase_done(mx, "text_encoder")

            denoiser = component("transformer")
            stage = time.perf_counter()
            flux = self.profile.family == "flux2-klein"
            rows, cols = request.height // 16, request.width // 16
            checkpoint = self._checkpoint
            channels = checkpoint.config("transformer")["in_channels"]
            dtype = mx.bfloat16 if flux else mx.float32
            noise_shape = (1, rows * cols, channels) if flux else (1, rows * 2, cols * 2, channels)
            latents = mx.random.normal(noise_shape, dtype=dtype, key=mx.random.key(request.seed))
            angles = denoiser.positions(rows, cols, prompt.shape[1])
            shift = checkpoint.config("scheduler")["shift"]
            sigmas = flow_schedule(self.profile.family, request.steps, rows * cols, z_shift=shift)
            step_seconds = []
            for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:], strict=True):
                _require_active(cancel)
                step_start = time.perf_counter()
                time_value = np.float32(sigma * 1000)
                if flux:
                    # Diffusers converts t to the latent dtype, divides, then rescales.
                    time_value = float((mx.array(time_value, dtype=dtype) / 1000 * 1000).item())
                else:
                    time_value = float((np.float32(1000) - time_value) / np.float32(1000))
                velocity = denoiser(latents, prompt, time_value, angles).astype(mx.float32)
                delta = float(next_sigma - sigma)
                latents = latents.astype(mx.float32) + delta * (velocity if flux else -velocity)
                latents = latents.astype(dtype)
                mx.eval(latents)
                step_seconds.append(time.perf_counter() - step_start)
            timings["denoising"] = time.perf_counter() - stage
            timings["denoiser_steps"] = step_seconds
            del denoiser, angles, prompt, velocity
            self._phase_done(mx, "transformer")

            decoder = component("vae")
            stage = time.perf_counter()
            rgb = decoder(latents, rows, cols).astype(mx.float32)
            mx.eval(rgb)
            pixels = np.array(rgb[0])
            if pixels.shape != (request.height, request.width, 3) or not np.isfinite(pixels).all():
                raise RuntimeError("image decoder produced invalid RGB data")
            timings["vae"] = time.perf_counter() - stage
            del decoder, rgb, latents
            self._phase_done(mx, "vae")

            _require_active(cancel)
            stage = time.perf_counter()
            encoded = io.BytesIO()
            pixels = np.rint(np.clip(pixels / 2 + 0.5, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(pixels).save(encoded, format="PNG")
            timings["png_encode"] = time.perf_counter() - stage
        except BaseException:
            mx.synchronize()
            self._release_weights()
            raise

        after = sample_telemetry()
        timings["total"] = time.perf_counter() - start
        return ImageResult(
            encoded.getvalue(),
            {
                "model": self.model,
                "revision": self.profile.artifact_revision(self.model),
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "denoiser_calls": len(step_seconds),
                "precision": self.profile.precision(self.quantization, published=self._published),
                "timings_seconds": timings,
                "memory": {
                    "mlx_peak_bytes": after.mlx_peak_bytes,
                    "mlx_active_bytes": after.mlx_active_bytes,
                    "mlx_cached_bytes": after.mlx_cached_bytes,
                    "rss_bytes": after.rss_bytes,
                    "host_pressure_before": before.os_pressure,
                    "host_pressure_after": after.os_pressure,
                    "host_swap_delta_bytes": (
                        after.swap_bytes - before.swap_bytes
                        if before.swap_bytes is not None and after.swap_bytes is not None
                        else None
                    ),
                },
            },
        )
