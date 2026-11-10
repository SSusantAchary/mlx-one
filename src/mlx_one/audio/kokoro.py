"""Safe native Kokoro loading, rule-based English phonemes, and waveform output."""

import hashlib
import json
import re
import threading
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from mlx_one.models.audio.kokoro.config import KokoroConfig
from mlx_one.models.audio.kokoro.weights import weight_contract
from mlx_one.retrieval.loading import _read_json, _read_safetensors

_FRONTEND_LOCK = threading.Lock()


@dataclass(frozen=True)
class LoadedKokoro:
    model: Any
    voice: Any
    path: Path
    model_id: str
    revision: str | None
    architecture: str = "kokoro"

    @property
    def auxiliary_arrays(self) -> dict[str, Any]:
        return {"voice": self.voice}


@dataclass(frozen=True)
class KokoroSpeech:
    waveform: np.ndarray
    phonemes: str
    durations: tuple[int, ...]
    model: str
    revision: str | None
    sample_rate: int = 24000
    voice: str = "af_heart"

    def to_dict(self) -> dict[str, Any]:
        return dict(
            model=self.model,
            revision=self.revision,
            voice=self.voice,
            sample_rate=self.sample_rate,
            samples=len(self.waveform),
            duration_seconds=len(self.waveform) / self.sample_rate,
            waveform_float32_sha256=hashlib.sha256(self.waveform.tobytes()).hexdigest(),
            phonemes=self.phonemes,
            durations=self.durations,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def save(self, path: str | Path) -> None:
        """Write mono PCM16 WAV; keep the unclipped native float waveform available."""
        pcm = (np.clip(self.waveform, -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(pcm.tobytes())


def load_kokoro_model(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    task: str = "synthesize",
) -> LoadedKokoro:
    """Load only a safely converted local Kokoro v1 bundle and af_heart voice."""
    if task not in {"synthesize", "speech-generation"}:
        raise ValueError("Kokoro implements speech synthesis only")
    root = Path(model).expanduser()
    if not root.is_dir():
        raise ValueError(
            "Kokoro publisher .pth weights require offline conversion; "
            "load the converted local directory"
        )
    data = _read_json(root / "config.json")
    if data.get("native_conversion") != "mlx-one-kokoro-v1":
        raise ValueError("Kokoro requires the explicit safe v1 conversion recipe")
    config = KokoroConfig.from_dict(data)
    tensors = _read_safetensors(root)
    weight_contract(config).validate(tensors)
    import mlx.core as mx

    from mlx_one.models.audio.kokoro.model import KokoroForSpeech

    native = KokoroForSpeech(config)
    native.load_weights(list(tensors.items()), strict=True)
    native.eval()
    voice = mx.load(str(root / "voices/af_heart.safetensors"))
    if set(voice) != {"voice"} or voice["voice"].shape != (510, 1, 256):
        raise ValueError("Kokoro af_heart voice assets do not match the supported profile")
    return LoadedKokoro(native, voice["voice"], root.resolve(), str(model), revision)


@lru_cache(maxsize=1)
def _english_backend():
    try:
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as exc:
        raise ValueError(
            "raw English text requires mlx-one[kokoro]; phoneme input works without it"
        ) from exc
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    return EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True, tie="^")


def phonemize_english(text: str) -> str:
    """Rule-based en-us eSpeak frontend with the publisher-compatible phoneme alphabet."""
    if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 8192:
        raise ValueError("English synthesis requires nonempty text within 8192 UTF-8 bytes")
    with _FRONTEND_LOCK:
        result = _english_backend().phonemize([text])[0].strip()
    pairs = {
        "ʔˌn\u0329": "ʔn",
        "ʔn\u0329": "ʔn",
        "a^ɪ": "I",
        "a^ʊ": "W",
        "d^ʒ": "ʤ",
        "e^ɪ": "A",
        "e": "A",
        "t^ʃ": "ʧ",
        "ɔ^ɪ": "Y",
        "ə^l": "ᵊl",
        "ʲo": "jo",
        "ʲə": "jə",
        "ʲ": "",
        "ɚ": "əɹ",
        "r": "ɹ",
        "x": "k",
        "ç": "k",
        "ɐ": "ə",
        "ɬ": "l",
        "\u0303": "",
    }
    for old in sorted(pairs, key=len, reverse=True):
        result = result.replace(old, pairs[old])
    result = re.sub(r"(\S)\u0329", r"ᵊ\1", result).replace(chr(809), "")
    for old, new in (
        ("o^ʊ", "O"),
        ("ɜːɹ", "ɜɹ"),
        ("ɜː", "ɜɹ"),
        ("ɪə", "iə"),
        ("ː", ""),
        ("o", "ɔ"),
        ("ɾ", "T"),
        ("ʔ", "t"),
        ("^", ""),
    ):
        result = result.replace(old, new)
    return result


def synthesize(
    model: str | Path | LoadedKokoro,
    text: str,
    *,
    voice: str = "af_heart",
    language: str = "en-us",
    phonemes: bool = False,
    speed: float = 1.0,
    revision: str | None = None,
    offline: bool = True,
) -> KokoroSpeech:
    """Synthesize a bounded English utterance with native MLX neural inference."""
    if voice != "af_heart" or language != "en-us":
        raise ValueError("the current Kokoro profile supports only af_heart and en-us")
    bundle = (
        model
        if isinstance(model, LoadedKokoro)
        else load_kokoro_model(model, revision=revision, offline=offline)
    )
    if type(phonemes) is not bool:
        raise ValueError("phonemes must be a boolean")
    sequence = text if phonemes else phonemize_english(text)
    if not isinstance(sequence, str) or not 1 <= len(sequence) <= 510:
        raise ValueError("Kokoro phoneme length must be between 1 and 510")
    output = bundle.model(sequence, bundle.voice[len(sequence) - 1], speed=speed)
    waveform = np.array(output.waveform).astype(np.float32)
    if waveform.ndim != 1 or not len(waveform) or not np.all(np.isfinite(waveform)):
        raise ValueError("Kokoro returned an invalid waveform")
    return KokoroSpeech(
        waveform, sequence, tuple(output.durations[0].tolist()), bundle.model_id, bundle.revision
    )
