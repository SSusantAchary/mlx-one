"""Generate small HF output fixtures; deterministic weights are reconstructed, not shipped."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    ViTConfig,
    ViTForImageClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
)


def array(name, shape):
    phase = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "little") / 65536
    values = np.sin(np.arange(np.prod(shape), dtype=np.float64) * 0.13 + phase) * 0.03
    if "norm" in name.lower() and name.endswith("weight") and len(shape) == 1:
        values += 1
    return values.reshape(shape).astype(np.float32)


def main():
    torch.set_num_threads(2)
    root = Path("tests/fixtures/batch1")
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        (
            "t5",
            T5Config(
                vocab_size=32,
                d_model=12,
                d_kv=4,
                d_ff=24,
                num_layers=2,
                num_decoder_layers=2,
                num_heads=3,
                dropout_rate=0,
                decoder_start_token_id=0,
                eos_token_id=1,
                pad_token_id=0,
            ),
            T5ForConditionalGeneration,
        ),
        (
            "vit",
            ViTConfig(
                image_size=8,
                patch_size=4,
                hidden_size=12,
                num_hidden_layers=2,
                num_attention_heads=3,
                intermediate_size=24,
                num_labels=3,
            ),
            ViTForImageClassification,
        ),
        (
            "xlm-embedding",
            XLMRobertaConfig(
                vocab_size=32,
                hidden_size=12,
                num_hidden_layers=2,
                num_attention_heads=3,
                intermediate_size=24,
                max_position_embeddings=18,
                pad_token_id=1,
                type_vocab_size=1,
            ),
            XLMRobertaModel,
        ),
        (
            "xlm-reranking",
            XLMRobertaConfig(
                vocab_size=32,
                hidden_size=12,
                num_hidden_layers=2,
                num_attention_heads=3,
                intermediate_size=24,
                max_position_embeddings=18,
                pad_token_id=1,
                type_vocab_size=1,
                num_labels=1,
            ),
            XLMRobertaForSequenceClassification,
        ),
    ]
    for name, config, factory in cases:
        kwargs = {"add_pooling_layer": False} if name == "xlm-embedding" else {}
        model = factory(config, **kwargs).eval()
        with torch.no_grad():
            for key, value in model.named_parameters():
                canonical = key.removeprefix("roberta.").replace(
                    ".attention.self.", ".attention.self_attn."
                )
                data = array(canonical, tuple(value.shape))
                value.copy_(torch.from_numpy(data))
        record = dict(
            config=config.to_dict(),
            transformers=transformers.__version__,
            torch=torch.__version__,
            weight_recipe="batch1_sine_weight_v1",
        )
        with torch.inference_mode():
            if name == "vit":
                pixels = array("pixels", (2, 3, 8, 8))
                result = model(torch.tensor(pixels), output_hidden_states=True)
                record.update(pixel_values=pixels.tolist(), logits=result.logits.tolist())
            else:
                ids = (
                    [[5, 6, 7, 0], [8, 9, 10, 11]]
                    if name == "t5"
                    else [[0, 5, 6, 2, 1], [0, 7, 8, 9, 2]]
                )
                mask = (
                    [[1, 1, 1, 0], [1, 1, 1, 1]]
                    if name == "t5"
                    else [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
                )
                record.update(input_ids=ids, attention_mask=mask)
                inputs = dict(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
                if name == "t5":
                    decoder = [[0, 12, 13], [0, 14, 15]]
                    result = model(**inputs, decoder_input_ids=torch.tensor(decoder))
                    record.update(
                        decoder_input_ids=decoder,
                        logits=result.logits.tolist(),
                        encoder_hidden=result.encoder_last_hidden_state.tolist(),
                        generation=model.generate(
                            **inputs, max_new_tokens=4, do_sample=False
                        ).tolist(),
                    )
                else:
                    result = model(**inputs, output_hidden_states=True)
                    record["hidden"] = result.hidden_states[-1].tolist()
                    if name == "xlm-reranking":
                        record["logits"] = result.logits.tolist()
        (root / f"{name}.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")
        print(name)


if __name__ == "__main__":
    main()
