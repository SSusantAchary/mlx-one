"""Native LightOnOCR-2 reusing Pixtral layers and MLX-One's Qwen3 decoder."""

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.language.qwen3.model import Qwen3ForCausalLM
from mlx_one.models.ocr._layers import InputEmbeddingsFeatures

from .projector import Mistral3MultiModalProjector
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.language_model = Qwen3ForCausalLM(config.text_config)

    def make_cache(self):
        from mlx_one.models.ocr._layers import KVCache

        return tuple(KVCache() for _ in range(self.config.text_config.num_hidden_layers))

    def get_input_embeddings(self, input_ids, pixel_values, image_sizes, **_):
        _, states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True, image_sizes=image_sizes
        )
        features = self.multi_modal_projector(states[-1][0], image_sizes)
        embeddings = self.language_model.model.embed_tokens(input_ids)
        positions = [
            i for i, t in enumerate(input_ids[0].tolist()) if t == self.config.image_token_index
        ]
        if len(positions) != features.shape[0]:
            raise ValueError("LightOnOCR image tokens do not match projected features")
        embeddings[0, mx.array(positions)] = features
        return InputEmbeddingsFeatures(inputs_embeds=embeddings)

    @staticmethod
    def sanitize(weights):
        result = {}
        for name, value in weights.items():
            if name.startswith("model.vision_encoder."):
                name = name.replace("model.vision_encoder.", "vision_tower.vision_model.", 1)
            elif name.startswith("model.vision_projection."):
                name = name.replace("model.vision_projection.", "multi_modal_projector.", 1)
            elif name.startswith("model.vision_tower."):
                name = name.removeprefix("model.")
            elif name.startswith("model.multi_modal_projector."):
                name = name.removeprefix("model.")
            elif name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "language_model.model.", 1)
            elif name.startswith("lm_head."):
                continue  # The pinned Qwen3 config ties the projection to embed_tokens.
            result[name] = value
        return result
