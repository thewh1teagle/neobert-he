"""ModernBERT MLM model for Hebrew pretraining."""

from __future__ import annotations

from transformers import ModernBertConfig, ModernBertForMaskedLM

from tokenization import build_vocab


def build_model(flash_attention: bool = False) -> ModernBertForMaskedLM:
    config = ModernBertConfig(
        vocab_size=len(build_vocab()),
        pad_token_id=0,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1536,
        max_position_embeddings=4096,
        attn_implementation="flash_attention_2" if flash_attention else "sdpa",
    )
    return ModernBertForMaskedLM(config)
