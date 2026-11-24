"""Fixed-depth Inkling text checkpoint configuration."""

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class InklingConfig:
    text: dict
    draft: dict
    precision: dict

    @classmethod
    def from_dict(cls, value):
        text = value["text_config"]
        loop = text["loop_config"]
        if (value.get("model_type") != "inkling2_loop"
                or text.get("model_type") != "inkling2_loop_text"
                or loop.get("is_varloop")
                or not 0 < loop["core_start"] < loop["core_end"] < text["num_hidden_layers"]
                or loop["num_passes"] < 1 or text["sliding_window_size"] < 1
                or text["num_attention_heads"] % text["num_key_value_heads"]
                or text["head_dim"] <= 0):
            raise ValueError("unsupported Inkling fixed-loop checkpoint")
        return cls(text, value["mtp_config"], value.get("mtplx_inkling_precision", {}))

    @property
    def text_config(self):
        return SimpleNamespace(**self.text, max_position_embeddings=self.max_position_embeddings)

    @property
    def max_position_embeddings(self):
        return self.text["model_max_length"]

    @property
    def visit_order(self):
        start, end, passes = (self.text["loop_config"][key]
                              for key in ("core_start", "core_end", "num_passes"))
        return (*range(start), *tuple(range(start, end)) * passes,
                *range(end, self.text["num_hidden_layers"]))
