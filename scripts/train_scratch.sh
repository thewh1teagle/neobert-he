#!/usr/bin/env bash
set -euo pipefail

uv run accelerate launch --mixed_precision fp16 src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/bert-char-he \
  --train-batch-size 64 \
  --epochs 3 \
  --encoder-lr 1e-4 \
  --warmup-steps 500 \
  --fp16 \
  "$@"
