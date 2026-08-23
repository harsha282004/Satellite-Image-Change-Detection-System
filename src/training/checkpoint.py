"""Checkpoint save/load. Every checkpoint stores enough to resume training exactly and to
reproduce which config/epoch/metric it came from (DEVELOPMENT_RULES.md Rule 7 — reproducibility)."""
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch: int, metrics: dict, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, map_location=None) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
