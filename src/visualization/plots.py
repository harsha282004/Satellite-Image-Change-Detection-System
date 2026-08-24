"""Training-curve plots from an experiment's history.csv (written by src/training/logger.py).

Run with: venv/Scripts/python.exe -m src.visualization.plots --experiment baseline_unet
"""
import argparse
import csv
from pathlib import Path


def load_history(experiment_name: str) -> list:
    path = Path("outputs/experiments") / experiment_name / "history.csv"
    if not path.exists():
        raise FileNotFoundError(f"No history.csv for experiment {experiment_name!r} at {path}")
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def plot_training_curves(experiment_name: str, out_path: Path = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_history(experiment_name)
    epochs = [r["epoch"] for r in rows]
    has_lr = bool(rows) and "lr" in rows[0]  # Phase 13+ experiments log lr; pre-Phase-13 ones don't

    n_rows = 3 if has_lr else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows))
    fig.suptitle(f"Training curves: {experiment_name}")

    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train_loss")
    ax.plot(epochs, [r["val_loss"] for r in rows], label="val_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [r["train_iou"] for r in rows], label="train_iou")
    ax.plot(epochs, [r["val_iou"] for r in rows], label="val_iou")
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU"); ax.set_title("IoU"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [r["train_dice"] for r in rows], label="train_dice")
    ax.plot(epochs, [r["val_dice"] for r in rows], label="val_dice")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice"); ax.set_title("Dice"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [r["val_precision"] for r in rows], label="val_precision")
    ax.plot(epochs, [r["val_recall"] for r in rows], label="val_recall")
    ax.plot(epochs, [r["val_f1"] for r in rows], label="val_f1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Validation Precision/Recall/F1")
    ax.legend(); ax.grid(alpha=0.3)

    if has_lr:
        ax = axes[2, 0]
        ax.plot(epochs, [r["lr"] for r in rows], label="lr", color="tab:red")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.set_title("Learning Rate (ReduceLROnPlateau)")
        ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
        axes[2, 1].axis("off")

    plt.tight_layout()

    if out_path is None:
        out_path = Path("outputs/visualizations") / f"{experiment_name}_training_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    out_path = plot_training_curves(args.experiment)
    print(f"Saved training curves: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
