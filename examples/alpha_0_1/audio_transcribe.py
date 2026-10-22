"""Offline native ASR candidate example; use a pinned local checkpoint and short audio."""
import argparse

from mlx_one import WhisperDecodeOptions, transcribe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local Whisper checkpoint directory")
    parser.add_argument("audio", help="Short local speech fixture")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = transcribe(args.model, args.audio, revision=args.revision, offline=True,
                        options=WhisperDecodeOptions(max_tokens=64, beam_size=1, best_of=1))
    print(result.text)


if __name__ == "__main__":
    main()
