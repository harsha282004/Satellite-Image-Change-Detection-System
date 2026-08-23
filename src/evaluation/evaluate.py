"""Test-set evaluation: loads a trained checkpoint, runs it on the held-out test split, reports
real measured IoU/Dice/Precision/Recall/F1/Accuracy, and saves a qualitative prediction grid
(before/after/ground-truth/prediction/overlay/diff) for visual inspection.

Reused across experiments (baseline now in Phase 4, Siamese variants later) rather than
duplicated per model, per DEVELOPMENT_RULES.md Rule 6.

Run with:
  venv/Scripts/python.exe -m src.evaluation.evaluate --config configs/baseline.yaml \
      --checkpoint outputs/checkpoints/baseline_unet/best.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.dataloader import get_dataloader
from src.evaluation.metrics import MetricAccumulator, logits_to_binary_preds
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model, load_config, resolve_device


@torch.no_grad()
def evaluate_test_set(model, test_loader, device) -> dict:
    model.eval()
    accumulator = MetricAccumulator()
    for before, after, mask in test_loader:
        before, after, mask = before.to(device), after.to(device), mask.to(device)
        logits = model(before, after)
        accumulator.update(logits, mask)
    return accumulator.compute()


@torch.no_grad()
def save_prediction_grid(model, dataset, device, n_samples: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3 * n_samples))
    col_titles = ["Before", "After", "Ground Truth", "Prediction", "Overlay (pred=red)", "Diff (FP=yellow, FN=blue)"]

    for row, idx in enumerate(indices):
        before, after, mask = dataset[idx]
        logits = model(before.unsqueeze(0).to(device), after.unsqueeze(0).to(device))
        pred = logits_to_binary_preds(logits)[0, 0].cpu().numpy()
        gt = mask[0].numpy()

        before_img = before.permute(1, 2, 0).numpy()
        after_img = after.permute(1, 2, 0).numpy()

        overlay = before_img.copy()
        overlay[pred > 0] = [1.0, 0.0, 0.0]

        diff = np.zeros((*gt.shape, 3), dtype=np.float32)
        diff[(pred == 1) & (gt == 0)] = [1.0, 1.0, 0.0]  # false positive
        diff[(pred == 0) & (gt == 1)] = [0.0, 0.3, 1.0]  # false negative
        diff[(pred == 1) & (gt == 1)] = [0.0, 1.0, 0.0]  # true positive

        panels = [before_img, after_img, gt, pred, overlay, diff]
        for col, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col] if n_samples > 1 else axes[col]
            cmap = "gray" if panel.ndim == 2 else None
            ax.imshow(panel, cmap=cmap)
            if row == 0:
                ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-viz-samples", type=int, default=6)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["device"])

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val metrics at save time: {checkpoint['metrics']}")

    test_loader = get_dataloader(
        root=config["dataset"]["root"],
        split="test",
        batch_size=config["dataloader"]["batch_size"],
        image_size=config["dataset"]["image_size"],
        num_workers=config["dataloader"]["num_workers"],
    )
    print(f"test samples: {len(test_loader.dataset)}, batches: {len(test_loader)}")

    metrics = evaluate_test_set(model, test_loader, device)
    print("\n=== TEST SET METRICS (real, measured) ===")
    for k in ("iou", "dice", "precision", "recall", "f1", "accuracy"):
        print(f"{k}: {metrics[k]:.4f}")

    experiment_name = config["experiment_name"]
    metrics_path = Path("outputs/metrics") / f"{experiment_name}_test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "experiment_name": experiment_name,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "test_metrics": metrics,
        }, f, indent=2)
    print(f"\nSaved test metrics: {metrics_path}")

    viz_path = Path("outputs/visualizations") / f"{experiment_name}_test_predictions.png"
    save_prediction_grid(model, test_loader.dataset, device, args.n_viz_samples, viz_path)
    print(f"Saved qualitative prediction grid: {viz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
