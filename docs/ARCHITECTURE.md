# Architecture

## Goal

Pretrain a Hebrew character-level encoder using Masked Language Modeling (MLM) on raw unvocalized Hebrew text. The pretrained encoder can then be loaded into a downstream G2P model.

## Model

`NeoBERTLMHead` in `src/model.py`:

1. **Encoder** — NeoBERT initialized from scratch (~20M params). 6 layers, 512 hidden, 8 heads, 2048 FFN, RoPE, SwiGLU, Pre-RMSNorm, 4096 context. See `src/encoder.py`.
2. **MLM head** — single linear layer: `hidden_size → vocab_size` (104 classes), weight-tied to the embedding.

Set `NEOBERT_ONNX_EXPORT=1` before importing to use ONNX-compatible ops instead of xformers kernels.

## Tokenizer

Character-level, 104-token vocab:
- 5 special tokens: `[PAD] [CLS] [SEP] [UNK] [MASK]`
- Hebrew letters א–ת (including final forms) + maqaf, geresh, gershayim
- ASCII lowercase, digits, punctuation, space

Built in-memory from `src/tokenization.py` — no external file needed.

## Training

See [TRAINING.md](TRAINING.md).
