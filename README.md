# bert-char-he

Hebrew character-level ModernBERT (~20M params) pretrained with MLM.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Quick Start

```console
# Split your Hebrew corpus
./scripts/split_dataset.sh /path/to/hebrew.txt

# Train
./scripts/train_scratch.sh

# Inference
uv run src/infer.py --checkpoint outputs/neobert-he/checkpoint-6500 --text "ש[MASK]ום ע[MASK]לם" # שלום עולם
```
