# Audio reference attribution

Native FastConformer, Depthformer and LFM waveform detokenizer components are
adapted from [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio/tree/76f35931cfe7db8f1f6b604d7b205ffe9d74ff8d/mlx_audio/sts/models/lfm_audio),
commit `76f35931cfe7db8f1f6b604d7b205ffe9d74ff8d`, MIT; retain UPSTREAM_LICENSE.
The loader, task pipeline and runtime integration use MLX-One's existing LFM2
backbone, hybrid caches, quantization and inference service. No mlx-audio runtime import.
Prompt/modality recipes were checked against Liquid4All/liquid-audio commit
`19e65845923a7f136442c95137884ec61eb386aa`. Publisher model terms remain separate.
