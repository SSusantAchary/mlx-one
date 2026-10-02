# LFM2.5 Audio

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `lfm2_audio`. The native architecture includes log-Mel preprocessing,
a Conformer audio encoder, audio-feature insertion into an LFM2 hybrid decoder,
Depthformer code predictions, and an audio detokenizer.
[LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)
is an upstream reference/candidate. Current validation is architectural and
synthetic; no complete native checkpoint task loader or registry weight
sanitizer/contract is registered for this entry.

## Installation and assets

Install `python -m pip install -e .` from the repository root using Python 3.10+.
Synthetic execution needs Apple Silicon/Metal. Future integration must provide
matching text/audio configs, complete backbone/Conformer/Depthformer/detokenizer
weights, tokenizer and modality-token metadata, processor settings, codec assets,
and publisher-specific prompt/decoding behavior. The registered architecture alone
does not supply that pipeline.

## Available inspection and configuration examples

```bash
mlx-one inspect LiquidAI/LFM2.5-Audio-1.5B
```

Validate a compatible nested config without initializing Metal; replace the path.
Publisher fields may require explicit mapping before this native schema accepts them.

```python
import json
from pathlib import Path

from mlx_one.models.audio.lfm2_audio.config import Lfm2AudioConfig

payload = json.loads(Path("/path/to/lfm2-audio/config.json").read_text())
config = Lfm2AudioConfig.from_dict(payload)
print(config.model_type, config.preprocessor_config.sampling_rate)
```

## Audio input and synthetic preprocessing

`waveform_to_log_mel` accepts `[samples]` or `[batch, samples]` arrays and returns
batched features. It does not infer the input sample rate; resample to the
configured rate first. This synthetic example verifies only the frontend API,
not publisher processor parity or intelligible speech generation.

```python
import mlx.core as mx

from mlx_one.models.audio.lfm2_audio.config import AudioPreprocessorConfig
from mlx_one.models.audio.lfm2_audio.processing import waveform_to_log_mel

config = AudioPreprocessorConfig()
waveform = mx.zeros((config.sampling_rate,), dtype=mx.float32)
features = waveform_to_log_mel(waveform, config)
mx.eval(features)
print(features.shape)
```

## Limits and troubleshooting

The model requires aligned audio features, token masks, and codebook dimensions.
Its default frontend uses 16-kHz samples and 128 Mel bins, subject to the actual
config. Conformer input width must match the frontend; Depthformer width and
detokenizer codebook count must match the nested configuration.

`mlx-one transcribe` and the public `transcribe` function execute Whisper, not
LFM2 Audio. There is no qualified file-to-transcript, speech-generation, or
audio-chat recipe for this family yet. Do not copy Whisper's processor or task
prompts to bypass the missing integration.

The LFM2 backbone retains convolution and attention state. Hybrid prefix restore
and block KV are unsupported; synthetic waveform output is not ASR/TTS quality
evidence. Shape errors require checking sample rate, Mel width, feature/token
alignment, and codec metadata.

## Implementation and validation

See [configuration](config.py), [model](model.py), [preprocessing](processing.py),
[architecture tests](../../../../../tests/test_native_lfm_architectures.py), and
[synthetic Metal tests](../../../../../tests/test_native_lfm_mlx.py).

```bash
pytest -q tests/test_native_lfm_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_lfm_mlx.py
```

See the [family backlog](../../../../../model_list.txt) and
[qualification levels](../../../../../README.md#capability-levels).
