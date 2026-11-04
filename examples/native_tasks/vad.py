"""Experimental offline and chunked mono16k VAD on exact local Silero weights."""

import argparse

from mlx_one import VADEngine
from mlx_one.audio.vad import prepare_vad

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
parser.add_argument("wav")
args = parser.parse_args()
pcm = prepare_vad(args.wav)
with VADEngine(args.checkpoint) as vad:
    offline = vad.detect(pcm)
    stream = vad.session()
    complete = len(pcm) // 512 * 512
    for offset in range(0, complete, 512):
        stream.process_chunk(pcm[offset : offset + 512])
    final = stream.finalize(pcm[complete:] if complete < len(pcm) else None)
    assert offline.to_dict() == final.to_dict()
    print(final.to_json())
