# P0 Engine model results

Measured source: `53404467afa3cf094bd221c61f5316f2b928b161`; working-source SHA256 `554ce34479cf89daf094caf9843c84a2b77eb1d45968e227fafd7d91f1ea68f6`.

Local evidence: `qualification/results/p0_engine/runs/1792201272601288000`. Checkpoints, raw traces and results are uncommitted.

## openai-community/gpt2

Status: **QUALIFIED**; revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

Architecture `gpt2`; logical parameters `124439808`; weight files `548105171` bytes.
Active text parameter count: `124439808`; tensor payload elements: `137022720`.
Native model class: `mlx_one.models.language.gpt2.model.GPT2LMHeadModel`.
Weight quantization: `{}`. Floating execution policy FP16; state dtypes are taken from evaluated components, not cache configuration labels.
Case counts: `{'PASS': 74, 'NOT_SUPPORTED': 9}`.

Numerical comparisons: 4; largest absolute logit error: 0.25. Original FP16/FP32 component tolerances apply; greedy agreement and state checks are recorded per case.
State categories: QuantizedKVCache._keys=float16; QuantizedKVCache._values=float16.

Warmed memory growth: `{"mlx_active_bytes": {"growth_bytes": 0, "limit_bytes": 33554432}, "rss_bytes": {"growth_bytes": 20889600, "limit_bytes": 134217728}}`.
GPT-2 scope is completion within 1024 total positions; chat and 2K/8K accepted-input profiles are inapplicable.

## mlx-community/Qwen3-0.6B-8bit

Status: **QUALIFIED**; revision `11de96878523501bcaa86104e3c186de07ff9068`.

Architecture `qwen3`; logical parameters `596049920`; weight files `633442994` bytes.
Active text parameter count: `596049920`; tensor payload elements: `167686144`.
Native model class: `mlx_one.models.language.qwen3.model.Qwen3ForCausalLM`.
Weight quantization: `{"bits": 8, "group_size": 64, "mode": "affine"}`. Floating execution policy FP16; state dtypes are taken from evaluated components, not cache configuration labels.
Case counts: `{'PASS': 83}`.

Numerical comparisons: 4; largest absolute logit error: 0.0. Original FP16/FP32 component tolerances apply; greedy agreement and state checks are recorded per case.
State categories: QuantizedKVCache._keys=float16; QuantizedKVCache._values=float16.

Warmed memory growth: `{"mlx_active_bytes": {"growth_bytes": 0, "limit_bytes": 33554432}, "rss_bytes": {"growth_bytes": 23953408, "limit_bytes": 134217728}}`.

## mlx-community/Qwen2.5-0.5B-Instruct-8bit

Status: **QUALIFIED**; revision `6474d4c1a05bd7bd634a9c993560e4e7272eda3d`.

Architecture `qwen2`; logical parameters `494032768`; weight files `525045902` bytes.
Active text parameter count: `494032768`; tensor payload elements: `138998144`.
Native model class: `mlx_one.models.language.qwen2.model.Qwen2ForCausalLM`.
Weight quantization: `{"bits": 8, "group_size": 64, "mode": "affine"}`. Floating execution policy FP16; state dtypes are taken from evaluated components, not cache configuration labels.
Case counts: `{'PASS': 83}`.

Numerical comparisons: 4; largest absolute logit error: 0.015625. Original FP16/FP32 component tolerances apply; greedy agreement and state checks are recorded per case.
State categories: QuantizedKVCache._keys=float16; QuantizedKVCache._values=float16.

Warmed memory growth: `{"mlx_active_bytes": {"growth_bytes": 0, "limit_bytes": 33554432}, "rss_bytes": {"growth_bytes": 38043648, "limit_bytes": 134217728}}`.

## mlx-community/LFM2-350M-4bit

Status: **QUALIFIED**; revision `18dc72abf3b2337f9123cfd6eeeb58dfa7947066`.

Architecture `lfm2`; logical parameters `354483968`; weight files `199527820` bytes.
Active text parameter count: `354483968`; tensor payload elements: `55443200`.
Native model class: `mlx_one.models.language.lfm2.model.Lfm2ForCausalLM`.
Weight quantization: `{"bits": 4, "group_size": 64, "mode": "affine"}`. Floating execution policy FP16; state dtypes are taken from evaluated components, not cache configuration labels.
Case counts: `{'PASS': 82, 'NOT_SUPPORTED': 1}`.

Numerical comparisons: 4; largest absolute logit error: 0.0. Original FP16/FP32 component tolerances apply; greedy agreement and state checks are recorded per case.
State categories: ConvCache.values=float16; QuantizedKVCache._keys=float16; QuantizedKVCache._values=float16.

Warmed memory growth: `{"mlx_active_bytes": {"growth_bytes": 0, "limit_bytes": 33554432}, "rss_bytes": {"growth_bytes": 12926976, "limit_bytes": 134217728}}`.
Ordinary complete hybrid state is tested; optional dense hybrid prefix restoration remains NOT_SUPPORTED and disabled.

## mlx-community/Qwen3.5-0.8B-4bit

Status: **QUALIFIED**; revision `da28692b5f139cb0ec58a356b437486b7dac7462`.

Architecture `qwen3_5`; logical parameters `852985920`; weight files `625229487` bytes.
Active text parameter count: `752393024`; tensor payload elements: `218574912`.
Native model class: `mlx_one.models.vision_language.qwen3_5.model.Qwen3_5ForConditionalGeneration`.
Weight quantization: `{"bits": 4, "group_size": 64, "mode": "affine"}`. Floating execution policy FP16; state dtypes are taken from evaluated components, not cache configuration labels.
Case counts: `{'PASS': 82, 'NOT_SUPPORTED': 1}`.

Numerical comparisons: 4; largest absolute logit error: 0.0. Original FP16/FP32 component tolerances apply; greedy agreement and state checks are recorded per case.
State categories: QuantizedKVCache._keys=float16; QuantizedKVCache._values=float16; Qwen3_5LinearCache.conv_state=float16; Qwen3_5LinearCache.recurrent_state=float32.

Warmed memory growth: `{"mlx_active_bytes": {"growth_bytes": 0, "limit_bytes": 33554432}, "rss_bytes": {"growth_bytes": 22380544, "limit_bytes": 134217728}}`.
Total loaded weights include vision/MTP components; the measured active-text parameter count is separate. Vision and MTP execution are outside this qualification.
Ordinary complete hybrid state is tested; optional dense hybrid prefix restoration remains NOT_SUPPORTED and disabled.
