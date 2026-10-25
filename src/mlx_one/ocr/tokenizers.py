"""Standalone tokenizers for original OCR checkpoints; never execute remote Python."""

import base64
import json
import unicodedata

from mlx_one.text.tokenizers import HFTokenizerAdapter


class GOTTokenizer:
    bos_token_id = 151643
    eos_token_id = 151643
    pad_token_id = 151643
    bos_token = eos_token = "<|endoftext|>"

    def __init__(self, root):
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "original GOT tokenization requires pip install 'mlx-one[ocr]'"
            ) from exc
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (
                line.split() for line in (root / "qwen.tiktoken").read_bytes().splitlines() if line
            )
        }
        names = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        names += [f"<|extra_{i}|>" for i in range(205)]
        names += [
            "<ref>",
            "</ref>",
            "<box>",
            "</box>",
            "<quad>",
            "</quad>",
            "<img>",
            "</img>",
            "<imgpad>",
        ]
        special = {name: len(ranks) + i for i, name in enumerate(names)}
        if special["<imgpad>"] != 151859:
            raise ValueError("GOT tokenizer vocabulary differs from the pinned image-token layout")
        self.special_tokens = special
        self.encoder = tiktoken.Encoding(
            name="mlx-one-got",
            mergeable_ranks=ranks,
            special_tokens=special,
            pat_str=(
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
                r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            ),
        )

    def encode(self, text, *, add_special_tokens=False):
        return self.encoder.encode(unicodedata.normalize("NFC", text), allowed_special="all")

    def decode(self, ids, *, skip_special_tokens=True):
        ids = list(ids)
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_tokens.values()]
        return self.encoder.decode(ids)


def load_tokenizer(root, family):
    if (root / "tokenizer.json").is_file():
        return HFTokenizerAdapter.from_directory(root)
    if family == "got_ocr2" and (root / "qwen.tiktoken").is_file():
        return GOTTokenizer(root)
    if family == "trocr" and (root / "vocab.json").is_file() and (root / "merges.txt").is_file():
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers

        vocab = json.loads((root / "vocab.json").read_text())
        merges = [
            tuple(line.split())
            for line in (root / "merges.txt").read_text().splitlines()
            if line and not line.startswith("#")
        ]
        tok = Tokenizer(models.BPE(vocab, merges))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        from tokenizers import AddedToken

        tok.add_special_tokens(
            [
                AddedToken(t, special=True)
                for t in ("<s>", "<pad>", "</s>", "<unk>", "<mask>")
                if t in vocab
            ]
        )
        return HFTokenizerAdapter(
            tok,
            bos_token="<s>",
            eos_token="</s>",
            bos_token_id=vocab["<s>"],
            eos_token_id=vocab["</s>"],
            pad_token_id=vocab["<pad>"],
            special_tokens={},
        )
    raise ValueError("checkpoint is missing tokenizer.json or supported original tokenizer assets")
