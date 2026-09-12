"""Built-in native architecture registrations."""

from mlx_one.core.registry import ModelRegistration, register_model


def register_builtin_models() -> None:
    registrations = (
        ModelRegistration(
            "gpt2",
            "mlx_one.models.language.gpt2.config:GPT2Config",
            "mlx_one.models.language.gpt2.model:GPT2LMHeadModel",
            "text",
            frozenset({"forward", "cache", "generate", "sample", "stream"}),
            "mlx_one.models.language.gpt2.weights:sanitize_weights",
            "mlx_one.models.language.gpt2.weights:weight_contract",
        ),
        ModelRegistration(
            "bert",
            "mlx_one.models.embeddings.bert.config:BertEmbeddingConfig",
            "mlx_one.models.embeddings.bert.model:BertForSentenceEmbedding",
            "embedding",
            frozenset({"forward", "mean-pooling", "normalize", "cosine"}),
        ),
        ModelRegistration(
            "lfm2",
            "mlx_one.models.language.lfm2.config:Lfm2Config",
            "mlx_one.models.language.lfm2.model:Lfm2ForCausalLM",
            "text",
            frozenset({"forward", "cache", "generate", "sample", "stream"}),
            "mlx_one.models.language.lfm2.weights:sanitize_weights",
            "mlx_one.models.language.lfm2.weights:weight_contract",
        ),
        ModelRegistration(
            "lfm2_moe",
            "mlx_one.models.language.lfm2_moe.config:Lfm2MoeConfig",
            "mlx_one.models.language.lfm2_moe.model:Lfm2MoeForCausalLM",
            "text",
            frozenset(
                {"forward", "cache", "router-outputs", "generate", "sample", "stream"}
            ),
            "mlx_one.models.language.lfm2_moe.weights:sanitize_weights",
            "mlx_one.models.language.lfm2_moe.weights:weight_contract",
        ),
        ModelRegistration(
            "lfm2_vl",
            "mlx_one.models.vision_language.lfm2_vl.config:Lfm2VLConfig",
            "mlx_one.models.vision_language.lfm2_vl.model:Lfm2VLForConditionalGeneration",
            "vision-language",
            frozenset({"forward", "cache", "image"}),
        ),
        ModelRegistration(
            "lfm2_colbert",
            "mlx_one.models.embeddings.lfm2_colbert.config:Lfm2ColBERTConfig",
            "mlx_one.models.embeddings.lfm2_colbert.model:Lfm2ColBERTModel",
            "embedding",
            frozenset({"forward", "late-interaction"}),
        ),
        ModelRegistration(
            "lfm2_audio",
            "mlx_one.models.audio.lfm2_audio.config:Lfm2AudioConfig",
            "mlx_one.models.audio.lfm2_audio.model:Lfm2AudioForConditionalGeneration",
            "audio-language",
            frozenset({"forward", "cache", "audio-encode", "audio-decode"}),
        ),
        ModelRegistration(
            "openelm",
            "mlx_one.models.language.openelm.config:OpenELMConfig",
            "mlx_one.models.language.openelm.model:OpenELMForCausalLM",
            "text",
            frozenset({"forward", "cache", "generate", "sample", "stream"}),
            "mlx_one.models.language.openelm.weights:sanitize_weights",
            "mlx_one.models.language.openelm.weights:weight_contract",
        ),
        ModelRegistration(
            "mpnet",
            "mlx_one.models.embeddings.mpnet.config:MPNetEmbeddingConfig",
            "mlx_one.models.embeddings.mpnet.model:MPNetForSentenceEmbedding",
            "embedding",
            frozenset({"forward", "mean-pooling", "normalize", "cosine"}),
        ),
        ModelRegistration(
            "qwen2",
            "mlx_one.models.language.qwen2.config:Qwen2Config",
            "mlx_one.models.language.qwen2.model:Qwen2ForCausalLM",
            "text",
            frozenset({"forward", "cache", "generate", "sample", "stream"}),
            "mlx_one.models.language.qwen2.weights:sanitize_weights",
            "mlx_one.models.language.qwen2.weights:weight_contract",
        ),
        ModelRegistration(
            "qwen3",
            "mlx_one.models.language.qwen3.config:Qwen3Config",
            "mlx_one.models.language.qwen3.model:Qwen3ForCausalLM",
            "text",
            frozenset({"forward", "cache", "generate", "sample", "stream"}),
            "mlx_one.models.language.qwen3.weights:sanitize_weights",
            "mlx_one.models.language.qwen3.weights:weight_contract",
        ),
        ModelRegistration(
            "qwen3_embedding",
            "mlx_one.models.embeddings.qwen3_embedding.config:Qwen3EmbeddingConfig",
            "mlx_one.models.embeddings.qwen3_embedding.model:Qwen3ForEmbedding",
            "embedding",
            frozenset({"forward", "last-token-pooling", "dimensions", "normalize", "cosine"}),
        ),
        ModelRegistration(
            "qwen3_reranker",
            "mlx_one.models.embeddings.qwen3_reranker.config:Qwen3RerankerConfig",
            "mlx_one.models.embeddings.qwen3_reranker.model:Qwen3ForReranking",
            "reranking",
            frozenset({"forward", "pair-scoring", "rerank", "probability"}),
        ),
        ModelRegistration(
            "qwen2_moe",
            "mlx_one.models.language.qwen2_moe.config:Qwen2MoeConfig",
            "mlx_one.models.language.qwen2_moe.model:Qwen2MoeForCausalLM",
            "text",
            frozenset(
                {"forward", "cache", "router-outputs", "generate", "sample", "stream"}
            ),
            "mlx_one.models.language.qwen2_moe.weights:sanitize_weights",
            "mlx_one.models.language.qwen2_moe.weights:weight_contract",
        ),
        ModelRegistration(
            "qwen2_vl",
            "mlx_one.models.vision_language.qwen2_vl.config:Qwen2VLConfig",
            "mlx_one.models.vision_language.qwen2_vl.model:Qwen2VLForConditionalGeneration",
            "vision-language",
            frozenset({"forward", "cache", "image", "video"}),
        ),
        ModelRegistration(
            "qwen2_5_vl",
            "mlx_one.models.vision_language.qwen2_5_vl.config:Qwen2_5_VLConfig",
            "mlx_one.models.vision_language.qwen2_5_vl.model:Qwen2_5_VLForConditionalGeneration",
            "vision-language",
            frozenset({"forward", "cache", "image", "video"}),
        ),
        ModelRegistration(
            "qwen3_5",
            "mlx_one.models.vision_language.qwen3_5.config:Qwen3_5Config",
            "mlx_one.models.vision_language.qwen3_5.model:Qwen3_5ForConditionalGeneration",
            "vision-language",
            frozenset(
                {
                    "forward",
                    "cache",
                    "image",
                    "video",
                    "linear-attention",
                    "generate",
                    "sample",
                    "stream",
                    "text-only",
                    "mtp",
                    "speculative-decoding",
                }
            ),
            "mlx_one.models.vision_language.qwen3_5.weights:sanitize_weights",
            "mlx_one.models.vision_language.qwen3_5.weights:weight_contract",
        ),
        ModelRegistration(
            "whisper",
            "mlx_one.models.audio.whisper.config:WhisperConfig",
            "mlx_one.models.audio.whisper.model:WhisperForConditionalGeneration",
            "asr",
            frozenset(
                {
                    "forward",
                    "cache",
                    "transcribe",
                    "language-detection",
                    "segment-timestamps",
                    "word-timestamps",
                }
            ),
        ),
    )
    for registration in registrations:
        register_model(registration)
