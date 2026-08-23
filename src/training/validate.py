"""Shared validation/test-set evaluation loop. Used both for per-epoch validation during
training and for the final held-out test-set evaluation in Phase 7 — one implementation, one
place the accumulation logic (MetricAccumulator, see src/evaluation/metrics.py) can be trusted."""
import torch

from src.evaluation.metrics import MetricAccumulator


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> tuple:
    model.eval()
    accumulator = MetricAccumulator()
    total_loss = 0.0
    n_batches = 0

    for before, after, mask in loader:
        before, after, mask = before.to(device), after.to(device), mask.to(device)
        logits = model(before, after)
        loss = loss_fn(logits, mask)

        total_loss += loss.item()
        n_batches += 1
        accumulator.update(logits, mask)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, accumulator.compute()
