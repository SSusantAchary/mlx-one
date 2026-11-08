"""Pinned PyLate-style ColBERT token preparation; offsets are not token features.

Recipe: prefix insertion after the initial token, right-padded query expansion,
separate attention/scoring masks, and document punctuation exclusion.
"""

from __future__ import annotations


def prepare_token_features(tokenizer, config, texts, *, input_type, max_length):
    query = input_type == "query"
    prefix = config.query_prefix if query else config.document_prefix
    limit = min(max_length, config.query_max_length if query else config.document_max_length)
    if limit < 2:
        raise ValueError("ColBERT prefix leaves no text capacity")
    backend = tokenizer._tokenizer
    prefix_id = backend.token_to_id(prefix)
    if prefix_id is None:
        raise ValueError("ColBERT prefix is absent from the checkpoint vocabulary")
    skip = {backend.token_to_id(word) for word in config.skiplist_words}
    skip.discard(None)
    sequences, attention, scoring = [], [], []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True, max_length=limit - 1)
        if not ids:
            raise ValueError("ColBERT requires an initial tokenizer token")
        ids = [ids[0], prefix_id, *ids[1:]]
        mask = [True] * len(ids)
        if query and config.do_query_expansion:
            pad = tokenizer.pad_token_id
            if pad is None:
                raise ValueError("ColBERT query expansion requires a pad token")
            padding = limit - len(ids)
            ids += [pad] * padding
            mask += [False] * padding
        if query and config.attend_to_expansion_tokens:
            mask = [True] * len(ids)
        keep = (
            [True] * len(ids)
            if query and config.do_query_expansion
            else list(mask)
            if query
            else [valid and token not in skip for token, valid in zip(ids, mask, strict=True)]
        )
        sequences.append(tuple(ids))
        attention.append(tuple(mask))
        scoring.append(tuple(keep))
    return tuple(sequences), tuple(attention), tuple(scoring)
