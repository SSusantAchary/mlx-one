# Native Kokoro English speech

ALBERT, duration/prosody networks, acoustic convolutions, harmonic excitation,
vocoder and FFT waveform synthesis execute in MLX. Native inference loads only
an explicitly converted safe bundle; the publisher PTH conversion is isolated
development tooling with pinned input hashes and a strict 587-tensor contract.

The admitted profile is `af_heart`, `en-us`, 24 kHz mono, speed 0.5–2 and at
most 510 phonemes / 30 seconds. Raw English requires the optional rule-based
frontend extra. Publisher-alphabet phoneme input needs no G2P dependency.

See the [model workflow guide](../../../../../docs/models/batch-1-sub-6b/README.md#kokoro-native-english-speech)
for conversion, Python/CLI/Engine/HTTP usage, the reviewed audio, source
attribution, measured RTF and unsupported voices/languages.
