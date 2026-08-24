"""Phase 15: decision-threshold sweep and selection.

`sweep_thresholds` runs the model over a dataloader exactly once (one forward pass per batch) and
reuses the resulting logits to compute metrics at every candidate threshold — cheaper than one
full forward pass per threshold, and guarantees every threshold's numbers come from the identical
set of model outputs.

Per PROJECT_CONTEXT.md / DEVELOPMENT_RULES.md Rule 3 (never use the test set for selection):
`select_best_threshold` is meant to be called on **validation** results only. The test set is
evaluated once, afterward, at the already-chosen threshold — see scripts/threshold_optimization.py.
"""
import torch

from src.evaluation.metrics import MetricAccumulator

DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


@torch.no_grad()
def sweep_thresholds(model, loader, device, thresholds=DEFAULT_THRESHOLDS) -> list:
    model.eval()
    accumulators = {t: MetricAccumulator(threshold=t) for t in thresholds}

    for before, after, mask in loader:
        before, after, mask = before.to(device), after.to(device), mask.to(device)
        logits = model(before, after)
        for t in thresholds:
            accumulators[t].update(logits, mask)

    results = []
    for t in thresholds:
        m = accumulators[t].compute()
        results.append({
            "threshold": t,
            "iou": m["iou"],
            "dice": m["dice"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    return results


def select_best_threshold(results: list, metric: str = "iou") -> float:
    """Picks the threshold with the highest value of `metric` among the swept results. Ties
    broken toward the threshold closest to 0.5 (the untuned default) — a mild, documented
    preference for the less-aggressive choice when multiple thresholds tie exactly."""
    best = max(results, key=lambda r: (r[metric], -abs(r["threshold"] - 0.5)))
    return best["threshold"]
