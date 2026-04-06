"""Train NeoBERT MLM on Hebrew text.

Example:
    uv run src/train.py \
        --train-dataset data/train.txt \
        --eval-dataset data/val.txt \
        --output-dir outputs/neobert-he-mlm

Multi-GPU:
    accelerate launch src/train.py \
        --train-dataset data/train.txt \
        --eval-dataset data/val.txt \
        --output-dir outputs/neobert-he-mlm
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from checkpoint import save_checkpoint, resume_step
from config import parse_args
from data import make_dataloaders
from eval import evaluate
from model import build_model
from optimizer import build_optimizer, build_scheduler
from tokenization import build_tokenizer


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=build_tokenizer(),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    train_loader, eval_loader = make_dataloaders(args, tokenizer)

    model = build_model(flash_attention=args.flash_attention)

    if args.init_from_checkpoint:
        from safetensors.torch import load_file
        state = load_file(str(Path(args.init_from_checkpoint) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.init_from_checkpoint}")

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    optimizer = build_optimizer(model, lr=args.encoder_lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_opt_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    opt_step = 0
    if args.init_from_checkpoint and not args.init_weights_only:
        opt_step = resume_step(args.init_from_checkpoint, scheduler)
        if accelerator.is_main_process and opt_step:
            print(f"Resumed from step {opt_step}")

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        epoch_loss_sum, epoch_steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True, disable=not accelerator.is_main_process)

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            with accelerator.autocast():
                out = model(**batch)

            scaled_loss = out.loss / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss_sum += out.loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / epoch_steps
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(step=opt_step, loss=f"{train_loss:.4f}", lr=f"{lr:.2e}")

                if accelerator.is_main_process:
                    if opt_step % args.logging_steps == 0:
                        print(f"[step {opt_step}] train_loss={train_loss:.4f} lr={lr:.2e}")
                        writer.add_scalar("train/loss", train_loss, opt_step)
                        writer.add_scalar("train/lr", lr, opt_step)

                    if opt_step % args.save_steps == 0:
                        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, accelerator)
                        writer.add_scalar("eval/loss", metrics["eval_loss"], opt_step)
                        writer.add_scalar("eval/acc", metrics["eval_acc"], opt_step)
                        print(f"[step {opt_step}] eval_loss={metrics['eval_loss']:.4f} eval_acc={metrics['eval_acc']:.4f}")
                        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["eval_loss"], args.save_total_limit)

    if accelerator.is_main_process:
        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, accelerator)
        print(f"Final eval_loss={metrics['eval_loss']:.4f} eval_acc={metrics['eval_acc']:.4f}")
        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["eval_loss"], args.save_total_limit)
        writer.close()


if __name__ == "__main__":
    main()
