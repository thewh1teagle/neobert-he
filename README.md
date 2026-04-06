# bert-char-he

Hebrew character-level BERT pretrained with MLM.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Quick Start

```console
# Split your Hebrew corpus
./scripts/split_dataset.sh /path/to/hebrew.txt

# Train
./scripts/train_scratch.sh

# Inference
uv run src/infer.py --checkpoint outputs/bert-char-he/checkpoint-1000 --text "מ[MASK]של[MASK]"
```
