"""Optimizer and LR schedule for MLM pretraining."""

from __future__ import annotations

import torch
from transformers import get_cosine_schedule_with_warmup


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.AdamW, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
