"""Backend-free helpers for the SAM3 configuration port."""

from __future__ import annotations

from copy import deepcopy
import logging as _logging
from types import SimpleNamespace

logging = SimpleNamespace(get_logger=_logging.getLogger)


def auto_docstring(obj=None, **kwargs):
    return obj if obj is not None else lambda value: value


class PreTrainedConfig:
    model_type = ""

    def __init__(self, **kwargs):
        self._attn_implementation = "sdpa"
        self.output_attentions = False
        self.output_hidden_states = False
        self.return_dict = True
        self.use_return_dict = True
        for key, value in kwargs.items():
            if key != "_attn_implementation":
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, payload):
        return cls(**deepcopy(payload))

    def to_dict(self):
        return {
            **{
                key: value.to_dict() if isinstance(value, PreTrainedConfig) else deepcopy(value)
                for key, value in vars(self).items()
                if not key.startswith("_")
            },
            "model_type": self.model_type,
        }


class CLIPTextConfig(PreTrainedConfig):
    model_type = "clip_text_model"

    def __init__(self, **kwargs):
        defaults = dict(
            vocab_size=49408,
            hidden_size=1024,
            intermediate_size=4096,
            projection_dim=512,
            num_hidden_layers=24,
            num_attention_heads=16,
            max_position_embeddings=32,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            eos_token_id=49407,
            bos_token_id=49406,
            pad_token_id=1,
            attention_dropout=0.0,
        )
        super().__init__(**(defaults | kwargs))


CONFIG_MAPPING = {}
AutoConfig = PreTrainedConfig
