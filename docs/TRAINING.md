# Training

## Commands

### 1. Prepare dataset

Split your corpus into train/val text files (one sentence per line), then tokenize and cache:

```console
uv run src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/neobert-he-mlm
```

The first run tokenizes and caches both files to Arrow (`.cache` directories next to each `.txt`). Subsequent runs reuse the cache.

### 2. Train

```console
uv run src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/my-run \
  --epochs 3 \
  --encoder-lr 2e-5 \
  --train-batch-size 16 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 200 \
  --logging-steps 50 \
  --save-steps 500
```

### Multi-GPU

```console
accelerate launch src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/my-run
```

### Fine-tuning from a checkpoint

Use `--init-from-checkpoint` to load weights only (resets optimizer state and step counter):

```console
uv run src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/my-run \
  --init-from-checkpoint outputs/previous-run/checkpoint-1200 \
  --init-weights-only \
  --epochs 1
```

## Flash Attention

Pass `--flash-attention` to use `flash_attention_2` instead of SDPA. Requires `flash-attn` installed:

```console
pip install flash-attn --no-build-isolation
```

```console
uv run src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/my-run \
  --flash-attention
```

## Key Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--encoder-lr` | `2e-5` | Learning rate |
| `--train-batch-size` | `16` | Batch size per device |
| `--gradient-accumulation-steps` | `1` | Effective batch multiplier |
| `--warmup-steps` | `200` | Linear warmup steps |
| `--fp16` | auto (CUDA) | Mixed precision training |
| `--flash-attention` | off | Enable flash attention + packing |
| `--init-weights-only` | off | Load weights but reset step/scheduler |

## Data Format

Input: plain `.txt` file, one Hebrew sentence per line. Nikud (diacritics) are stripped automatically by the tokenizer normalizer — no preprocessing needed.

## Data Pipeline

```
raw .txt (one sentence per line)
  → data.py                tokenize + cache to Arrow (once)
  → DataCollatorForMLM     15% random masking per batch
  → train.py               training loop
```

## Using the Pretrained Encoder

Load encoder weights into a downstream model:

```console
uv run src/train.py \
  --init-from-checkpoint outputs/neobert-he/checkpoint-XXXX \
  --init-weights-only \
  ...
```
