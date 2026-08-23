"""Training loop orchestration: train one epoch, validate, checkpoint the best model by
validation IoU, and log every epoch's metrics. Model-agnostic — works with any model whose
forward signature is `model(before, after) -> logits`, so it's reused unchanged for the Siamese
U-Net in Phase 5 (DEVELOPMENT_RULES.md Rule 6 — don't rewrite working modules)."""
from pathlib import Path

from src.evaluation.metrics import MetricAccumulator
from src.training.checkpoint import save_checkpoint
from src.training.logger import ExperimentLogger
from src.training.validate import validate


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, checkpoint_dir, experiment_name, config: dict):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config

        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(Path("outputs/experiments") / experiment_name)

        self.best_val_iou = -1.0

    def train_one_epoch(self, loader) -> tuple:
        self.model.train()
        accumulator = MetricAccumulator()
        total_loss = 0.0
        n_batches = 0

        for before, after, mask in loader:
            before, after, mask = before.to(self.device), after.to(self.device), mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(before, after)
            loss = self.loss_fn(logits, mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            accumulator.update(logits.detach(), mask)

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, accumulator.compute()

    def fit(self, train_loader, val_loader, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_one_epoch(train_loader)
            val_loss, val_metrics = validate(self.model, val_loader, self.loss_fn, self.device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_loss,
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
            self.logger.log_epoch(row)
            print(
                f"[epoch {epoch}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_iou={train_metrics['iou']:.4f} | "
                f"val_loss={val_loss:.4f} val_iou={val_metrics['iou']:.4f} "
                f"val_dice={val_metrics['dice']:.4f} val_f1={val_metrics['f1']:.4f}"
            )

            save_checkpoint(
                self.checkpoint_dir / "last.pt", self.model, self.optimizer, epoch, val_metrics, self.config
            )
            if val_metrics["iou"] > self.best_val_iou:
                self.best_val_iou = val_metrics["iou"]
                save_checkpoint(
                    self.checkpoint_dir / "best.pt", self.model, self.optimizer, epoch, val_metrics, self.config
                )
                print(f"  -> new best checkpoint saved (val_iou={val_metrics['iou']:.4f})")

        self.logger.save_json()
        self.logger.close()
        return self.best_val_iou
