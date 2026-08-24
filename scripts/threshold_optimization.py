"""Phase 15.2: threshold optimization on the real best model.

Sweeps thresholds 0.30-0.70 (step 0.05) on the VALIDATION set only, selects the best threshold by
validation IoU, then evaluates that single chosen threshold once on the TEST set — the test set is
never used to pick the threshold (DEVELOPMENT_RULES.md Rule 3 / PROJECT_CONTEXT.md).

Run with: venv/Scripts/python.exe scripts/threshold_optimization.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataloader import get_dataloader
from src.evaluation.metrics import MetricAccumulator
from src.evaluation.threshold_analysis import DEFAULT_THRESHOLDS, select_best_threshold, sweep_thresholds
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model, load_config, resolve_device

CONFIG = "configs/siamese_attention_e100.yaml"
CHECKPOINT = "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt"


def save_threshold_plot(results: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in results]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = ["iou", "dice", "precision", "recall", "f1"]
    titles = ["IoU vs Threshold", "Dice vs Threshold", "Precision vs Threshold",
              "Recall vs Threshold", "F1 vs Threshold"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 3, i % 3]
        ax.plot(thresholds, [r[metric] for r in results], marker="o", color="tab:blue")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    config = load_config(CONFIG)
    device = resolve_device(config["device"])
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(CHECKPOINT, model, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    val_loader = get_dataloader(
        root=config["dataset"]["root"], split="val", batch_size=config["dataloader"]["batch_size"],
        image_size=config["dataset"]["image_size"], num_workers=0,
    )
    test_loader = get_dataloader(
        root=config["dataset"]["root"], split="test", batch_size=config["dataloader"]["batch_size"],
        image_size=config["dataset"]["image_size"], num_workers=0,
    )

    print(f"\n=== Sweeping {len(DEFAULT_THRESHOLDS)} thresholds on VALIDATION set "
          f"({len(val_loader.dataset)} images) ===")
    val_results = sweep_thresholds(model, val_loader, device, thresholds=DEFAULT_THRESHOLDS)
    for r in val_results:
        print(f"  threshold={r['threshold']:.2f}  IoU={r['iou']:.4f}  Dice={r['dice']:.4f}  "
              f"Precision={r['precision']:.4f}  Recall={r['recall']:.4f}  F1={r['f1']:.4f}")

    best_threshold = select_best_threshold(val_results, metric="iou")
    default_row = next(r for r in val_results if abs(r["threshold"] - 0.5) < 1e-9)
    best_row = next(r for r in val_results if abs(r["threshold"] - best_threshold) < 1e-9)
    print(f"\nSelected threshold (by validation IoU): {best_threshold:.2f} "
          f"(val IoU={best_row['iou']:.4f} vs. default 0.50's val IoU={default_row['iou']:.4f})")

    csv_path = Path("outputs/metrics/threshold_analysis.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "iou", "dice", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(val_results)
    print(f"Saved: {csv_path}")

    save_threshold_plot(val_results, Path("outputs/visualizations/threshold_analysis.png"))
    print("Saved: outputs/visualizations/threshold_analysis.png")

    print(f"\n=== Evaluating selected threshold ({best_threshold:.2f}) ONCE on TEST set "
          f"({len(test_loader.dataset)} images) — never used for threshold selection ===")
    test_accumulator = MetricAccumulator(threshold=best_threshold)
    import torch
    model.eval()
    with torch.no_grad():
        for before, after, mask in test_loader:
            before, after, mask = before.to(device), after.to(device), mask.to(device)
            logits = model(before, after)
            test_accumulator.update(logits, mask)
    test_metrics = test_accumulator.compute()

    # Also compute the default-0.5 test result for a direct, honest comparison.
    default_accumulator = MetricAccumulator(threshold=0.5)
    with torch.no_grad():
        for before, after, mask in test_loader:
            before, after, mask = before.to(device), after.to(device), mask.to(device)
            logits = model(before, after)
            default_accumulator.update(logits, mask)
    default_test_metrics = default_accumulator.compute()

    print(f"\nTest @ selected threshold {best_threshold:.2f}: "
          f"IoU={test_metrics['iou']:.4f} Dice={test_metrics['dice']:.4f} "
          f"Precision={test_metrics['precision']:.4f} Recall={test_metrics['recall']:.4f} "
          f"F1={test_metrics['f1']:.4f} Accuracy={test_metrics['accuracy']:.4f}")
    print(f"Test @ default threshold 0.50:    "
          f"IoU={default_test_metrics['iou']:.4f} Dice={default_test_metrics['dice']:.4f} "
          f"Precision={default_test_metrics['precision']:.4f} Recall={default_test_metrics['recall']:.4f} "
          f"F1={default_test_metrics['f1']:.4f} Accuracy={default_test_metrics['accuracy']:.4f}")

    report = {
        "checkpoint": CHECKPOINT,
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_sweep": val_results,
        "selected_threshold": best_threshold,
        "selection_criterion": "max validation IoU",
        "test_metrics_at_selected_threshold": test_metrics,
        "test_metrics_at_default_threshold_0.5": default_test_metrics,
    }
    report_path = Path("outputs/metrics/threshold_optimization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
