# bert-char-he

Hebrew character-level BERT (~20M params) pretrained with MLM on raw unvocalized Hebrew text.

## Features

- **Character-level** — 104-token vocab (Hebrew letters, ASCII, punctuation). No subword tokenization.
- **NeoBERT architecture** — RoPE embeddings, SwiGLU activation, Pre-RMSNorm, full attention every layer
- **ONNX exportable** — set `NEOBERT_ONNX_EXPORT=1` to switch to ONNX-compatible ops
- **Compact** — 6 layers, 512 hidden, 8 heads, 2048 FFN, 4096 token context
- **Downstream-ready** — used as encoder in [renikud](https://github.com/thewh1teagle/renikud) for Hebrew G2P

## Usage

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("thewh1teagle/bert-char-he")
model = AutoModel.from_pretrained("thewh1teagle/bert-char-he")
```

## Quick Start

```console
# Split your Hebrew corpus
./scripts/split_dataset.sh /path/to/hebrew.txt

# Train
./scripts/train_scratch.sh

# Inference
uv run src/infer.py --checkpoint outputs/bert-char-he/checkpoint-6500 --text "ש[MASK]ום ע[MASK]לם"
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.
