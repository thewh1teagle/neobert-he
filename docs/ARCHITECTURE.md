# Architecture

## Goal

Pretrain a Hebrew character-level encoder using Masked Language Modeling (MLM) on raw unvocalized Hebrew text. The pretrained encoder can then be loaded into a downstream G2P model.

## Model

`ModernBertForMaskedLM` in `src/model.py`:

1. **Encoder** — ModernBERT initialized from scratch (~20M params). 6 layers, 512 hidden, 8 heads, 1536 FFN, RoPE, SDPA. See `src/encoder.py`.
2. **MLM head** — single linear layer: `hidden_size → vocab_size` (104 classes), weight-tied to the embedding.

## Tokenizer

Character-level, 104-token vocab:
- 5 special tokens: `[PAD] [CLS] [SEP] [UNK] [MASK]`
- Hebrew letters א–ת (including final forms) + maqaf, geresh, gershayim
- ASCII lowercase, digits, punctuation, space

Built in-memory from `src/tokenization.py` — no external file needed.

## Training

See [TRAINING.md](TRAINING.md).
