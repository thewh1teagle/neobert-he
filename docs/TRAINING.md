# Training

## 1. Prepare data

```console
bash scripts/prepare_dataset.sh
bash scripts/split_dataset.sh data/raw.txt
```

This downloads the Hebrew corpora, strips nikud, and splits into `data/train.txt` / `data/val.txt`.

## 2. Train

```console
bash scripts/train_scratch.sh
```

Checkpoints are saved to `outputs/neobert-he/` in HuggingFace format, loadable with `AutoModelForMaskedLM.from_pretrained`.

### Resume from checkpoint

```console
bash scripts/train_scratch.sh --init-from-checkpoint outputs/neobert-he/checkpoint-2000
```

### Fine-tune from a checkpoint (reset optimizer)

```console
bash scripts/train_scratch.sh --init-from-checkpoint outputs/neobert-he/checkpoint-2000 --init-weights-only
```

## Flash Attention

ModernBERT supports Flash Attention 2 for faster training and lower VRAM usage. Enable with `--flash-attention`:

```console
bash scripts/train_scratch.sh --flash-attention
```

Install a prebuilt wheel first:

- **x86_64**: https://github.com/mjun0812/flash-attention-prebuild-wheels
- **aarch64 (ARM)**: https://pypi.jetson-ai-lab.io/sbsa/cu130

Validate:

```console
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

## Data notes

- Input: one Hebrew sentence per line
- Nikud (diacritics) are stripped automatically — no preprocessing needed
- Masking: 15% of characters masked per batch (MLM)
