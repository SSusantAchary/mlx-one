"""Optional native Whisper round-trip diagnostic; does not replace listening."""

import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np

from mlx_one.audio import transcribe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with wave.open(str(args.audio)) as source:
        assert source.getnchannels() == 1 and source.getsampwidth() == 2
        rate = source.getframerate()
        pcm = (
            np.frombuffer(source.readframes(source.getnframes()), "<i2").astype(np.float32) / 32768
        )
    resampled = np.interp(
        np.arange(int(len(pcm) * 16000 / rate)) * rate / 16000, np.arange(len(pcm)), pcm
    ).astype(np.float32)
    result = transcribe(args.model, resampled, sample_rate=16000, language="en", offline=True)
    expected = "Hello, world. This is native speech synthesis on Apple Silicon."

    def normalize(text):
        return re.findall(r"[a-z]+", text.lower())

    reference, actual = normalize(expected), normalize(result.text)
    costs = list(range(len(actual) + 1))
    for i, word in enumerate(reference, 1):
        row = [i]
        for j, candidate in enumerate(actual, 1):
            row.append(min(row[-1] + 1, costs[j] + 1, costs[j - 1] + (word != candidate)))
        costs = row
    args.output.write_text(
        json.dumps(
            dict(
                checkpoint="openai/whisper-tiny",
                revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                audio_sha256=hashlib.sha256(args.audio.read_bytes()).hexdigest(),
                reference=expected,
                transcript=result.text,
                reference_words=len(reference),
                word_errors=costs[-1],
                wer=costs[-1] / len(reference),
                status="MEASURED",
                normalization="lowercase a-z word tokens",
                resampling="explicit NumPy linear interpolation from 24 kHz to 16 kHz",
                scope="one native TTS-to-native ASR sample; not a corpus or human listening result",
            ),
            indent=2,
        )
        + "\n"
    )
    print(result.text)


if __name__ == "__main__":
    main()
