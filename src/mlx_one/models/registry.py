"""Built-in native architecture registrations."""

from mlx_one.core.registry import ModelRegistration, register_model


def register_builtin_models() -> None:
    registrations = (
        ModelRegistration(
            "openelm",
            "mlx_one.models.language.openelm.config:OpenELMConfig",
            "mlx_one.models.language.openelm.model:OpenELMForCausalLM",
            "text",
            frozenset({"forward", "cache"}),
        ),
        ModelRegistration(
            "qwen2",
            "mlx_one.models.language.qwen2.config:Qwen2Config",
            "mlx_one.models.language.qwen2.model:Qwen2ForCausalLM",
            "text",
            frozenset({"forward", "cache"}),
        ),
        ModelRegistration(
            "qwen3",
            "mlx_one.models.language.qwen3.config:Qwen3Config",
            "mlx_one.models.language.qwen3.model:Qwen3ForCausalLM",
            "text",
            frozenset({"forward", "cache"}),
        ),
        ModelRegistration(
            "qwen2_moe",
            "mlx_one.models.language.qwen2_moe.config:Qwen2MoeConfig",
            "mlx_one.models.language.qwen2_moe.model:Qwen2MoeForCausalLM",
            "text",
            frozenset({"forward", "cache", "router-outputs"}),
        ),
        ModelRegistration(
            "qwen2_vl",
            "mlx_one.models.vision_language.qwen2_vl.config:Qwen2VLConfig",
            "mlx_one.models.vision_language.qwen2_vl.model:Qwen2VLForConditionalGeneration",
            "vision-language",
            frozenset({"forward", "cache", "image", "video"}),
        ),
    )
    for registration in registrations:
        register_model(registration)
