"""Segmentation metrics for binary change masks: IoU, Dice, Precision, Recall, F1, Accuracy.

Per PROJECT_CONTEXT.md, accuracy alone is not a sufficient metric on this class-imbalanced task
(Phase 2 measured ~4-5% changed pixels), so all five are always reported together.

`MetricAccumulator` accumulates confusion-matrix counts (TP/FP/FN/TN) across an entire epoch/split
before computing ratios, rather than averaging per-batch metrics — the correct approach under
class imbalance, since a per-batch average would let batches with few/no changed pixels skew the
result.
"""
import torch


def logits_to_binary_preds(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.reshape(-1).bool()
    targets = targets.reshape(-1).bool()

    tp = (preds & targets).sum().item()
    fp = (preds & ~targets).sum().item()
    fn = (~preds & targets).sum().item()
    tn = (~preds & ~targets).sum().item()
    return tp, fp, fn, tn


class MetricAccumulator:
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.tp = self.fp = self.fn = self.tn = 0

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits_to_binary_preds(logits, self.threshold)
        tp, fp, fn, tn = confusion_counts(preds, targets)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute(self) -> dict:
        eps = self.eps
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
