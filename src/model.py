"""NeoBERT MLM model for Hebrew pretraining."""

from __future__ import annotations

from neobert.model import NeoBERTLMHead, NeoBERTConfig

from tokenization import build_vocab


def _vocab_size() -> int:
    return len(build_vocab())


def build_model(flash_attention: bool = False) -> NeoBERTLMHead:
    config = NeoBERTConfig(
        vocab_size=_vocab_size(),
        num_hidden_layers=6,
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        max_length=4096,
    )
    return NeoBERTLMHead(config)
