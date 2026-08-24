"""Loss functions for binary change-mask segmentation. Operate on raw logits (not sigmoid
probabilities) for numerical stability, matching model outputs from models/unet.py.
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and Dice loss — the default baseline/Siamese loss,
    since Dice alone can be unstable early in training on heavily imbalanced masks."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017): down-weights easy (already well-classified) pixels
    so the loss focuses on hard/misclassified ones — a different response to class imbalance than
    Dice's set-overlap formulation. `alpha` weights the positive (changed-pixel) class; `gamma`
    controls how strongly easy examples are down-weighted (gamma=0 reduces to weighted BCE)."""

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # = p if target==1 else (1-p), recovered stably from the BCE value
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal_term.mean()


class FocalDiceLoss(nn.Module):
    """Focal loss + Dice loss, analogous to BCEDiceLoss but with Focal in place of plain BCE —
    tests whether focusing training on hard pixels (rather than just re-weighting BCE by class)
    helps on this task's ~4-5% positive-pixel imbalance (docs/DATASET.md)."""

    def __init__(self, focal_alpha: float = 0.8, focal_gamma: float = 2.0,
                 focal_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)


class WeightedBCEDiceLoss(nn.Module):
    """BCEDiceLoss with a positive-class weight on the BCE term (`pos_weight` in
    `nn.BCEWithLogitsLoss`, applied to the minority "changed" class). Default 5.0 is a moderate,
    documented choice — not the exact inverse class ratio (~1:21 changed:unchanged pixels per
    docs/DATASET.md, which would suggest ~20) — chosen because very large pos_weight values are
    known to hurt precision by over-predicting the positive class; this is an experiment input to
    test empirically, not assumed correct."""

    def __init__(self, pos_weight: float = 5.0, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight_value = pos_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor(self.pos_weight_value, device=logits.device, dtype=logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


class TverskyLoss(nn.Module):
    """Tversky index generalizes Dice with independent false-positive/false-negative weights:
    TI = TP / (TP + alpha*FP + beta*FN); loss = 1 - TI. alpha=beta=0.5 reduces exactly to Dice.
    Default alpha=0.3, beta=0.7 (Salehi et al., 2017) penalizes false negatives more than false
    positives — i.e. favors recall — a direct, configurable lever on the precision/recall tradeoff
    observed between models throughout this project (docs/EVALUATION.md, docs/EXPERIMENTS.md)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=1)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


def get_loss(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss(**kwargs)
    if name in ("bce_dice", "bcedice"):
        return BCEDiceLoss(**kwargs)
    if name in ("focal_dice", "focaldice"):
        return FocalDiceLoss(**kwargs)
    if name in ("weighted_bce_dice", "weightedbcedice"):
        return WeightedBCEDiceLoss(**kwargs)
    if name == "tversky":
        return TverskyLoss(**kwargs)
    raise ValueError(
        f"Unknown loss name: {name!r} (expected one of 'bce', 'dice', 'bce_dice', 'focal_dice', "
        f"'weighted_bce_dice', 'tversky')"
    )
