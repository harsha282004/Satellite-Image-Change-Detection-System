"""Training loop orchestration: train one epoch, validate, checkpoint the best model by
validation IoU, and log every epoch's metrics. Model-agnostic — works with any model whose
forward signature is `model(before, after) -> logits`, so it's reused unchanged for the Siamese
U-Net in Phase 5 (DEVELOPMENT_RULES.md Rule 6 — don't rewrite working modules).

Phase 13 adds optional early stopping and an optional LR scheduler, both off by default so every
config written before Phase 13 (baseline.yaml, siamese.yaml, siamese_diff.yaml, siamese_concat.yaml,
siamese_attention.yaml) trains exactly as it did before — full fixed epoch count, constant LR,
identical checkpoints/history if re-run. Only new configs that explicitly set `training.
early_stopping`/`training.scheduler` opt into the new behavior.
"""
from pathlib import Path

from src.evaluation.metrics import MetricAccumulator
from src.training.checkpoint import save_checkpoint
from src.training.logger import ExperimentLogger
from src.training.validate import validate


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, checkpoint_dir, experiment_name, config: dict,
                 scheduler=None, early_stopping: dict = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.scheduler = scheduler

        es = early_stopping or {}
        self.early_stopping_enabled = bool(es.get("enabled", False))
        self.early_stopping_patience = int(es.get("patience", 10))
        self.early_stopping_monitor = es.get("monitor", "val_iou")

        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(Path("outputs/experiments") / experiment_name)

        self.best_val_iou = -1.0
        self.best_epoch = 0

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

    def fit(self, train_loader, val_loader, num_epochs: int) -> dict:
        epochs_no_improve = 0
        early_stopped = False
        epochs_trained = 0

        for epoch in range(1, num_epochs + 1):
            epochs_trained = epoch
            train_loss, train_metrics = self.train_one_epoch(train_loader)
            val_loss, val_metrics = validate(self.model, val_loader, self.loss_fn, self.device)
            current_lr = self.optimizer.param_groups[0]["lr"]

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
                "lr": current_lr,
            }
            self.logger.log_epoch(row)
            print(
                f"[epoch {epoch}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_iou={train_metrics['iou']:.4f} | "
                f"val_loss={val_loss:.4f} val_iou={val_metrics['iou']:.4f} "
                f"val_dice={val_metrics['dice']:.4f} val_f1={val_metrics['f1']:.4f} | "
                f"lr={current_lr:.2e}"
            )

            save_checkpoint(
                self.checkpoint_dir / "last.pt", self.model, self.optimizer, epoch, val_metrics, self.config
            )

            improved = val_metrics["iou"] > self.best_val_iou
            if improved:
                self.best_val_iou = val_metrics["iou"]
                self.best_epoch = epoch
                epochs_no_improve = 0
                save_checkpoint(
                    self.checkpoint_dir / "best.pt", self.model, self.optimizer, epoch, val_metrics, self.config
                )
                print(f"  -> new best checkpoint saved (val_iou={val_metrics['iou']:.4f})")
            else:
                epochs_no_improve += 1

            if self.scheduler is not None:
                monitor_value = val_metrics["iou"] if self.early_stopping_monitor == "val_iou" else val_loss
                self.scheduler.step(monitor_value)

            if self.early_stopping_enabled and epochs_no_improve >= self.early_stopping_patience:
                early_stopped = True
                print(
                    f"  -> early stopping: no improvement in {self.early_stopping_monitor} for "
                    f"{epochs_no_improve} epochs (patience={self.early_stopping_patience})"
                )
                break

        self.logger.save_json()
        self.logger.close()

        return {
            "best_val_iou": self.best_val_iou,
            "best_epoch": self.best_epoch,
            "max_epochs": num_epochs,
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
        }
