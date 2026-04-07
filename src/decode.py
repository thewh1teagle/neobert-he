"""Decode token IDs back to characters for debugging and inference."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"}


def ids_to_str(ids: list[int], tokenizer: PreTrainedTokenizerFast, skip_special: bool = True) -> str:
    """Convert a list of token IDs to a string, optionally skipping special tokens."""
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if skip_special:
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
    return "".join(tokens)


def decode_masked_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    n: int = 4,
) -> list[dict[str, str]]:
    """Return up to n examples from a batch showing input, prediction, and ground truth.

    Each example is a dict with keys: input, pred, true.
    Only masked positions (labels != -100) are shown as differing from the input.
    """
    preds = logits.argmax(-1)  # [B, S]
    results = []

    for i in range(min(n, input_ids.size(0))):
        ids = input_ids[i].tolist()
        lab = labels[i].tolist()
        pred = preds[i].tolist()

        # Build strings: replace masked positions with pred/true tokens
        pred_ids = [pred[j] if lab[j] != -100 else ids[j] for j in range(len(ids))]
        true_ids = [lab[j] if lab[j] != -100 else ids[j] for j in range(len(ids))]

        results.append({
            "input": ids_to_str(ids, tokenizer),
            "pred":  ids_to_str(pred_ids, tokenizer),
            "true":  ids_to_str(true_ids, tokenizer),
        })

    return results
