"""Phase 15: threshold sweep/selection correctness, using a tiny synthetic model+loader."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.siamese_unet import SiameseUNet
from src.evaluation.threshold_analysis import select_best_threshold, sweep_thresholds


def _make_loader(n=4, size=16):
    before = torch.rand(n, 3, size, size)
    after = torch.rand(n, 3, size, size)
    mask = torch.randint(0, 2, (n, 1, size, size)).float()
    return DataLoader(TensorDataset(before, after, mask), batch_size=2)


def test_sweep_thresholds_returns_one_row_per_threshold_with_expected_keys():
    model = SiameseUNet(base_channels=4, comparison_mode="diff")
    thresholds = (0.3, 0.5, 0.7)
    results = sweep_thresholds(model, _make_loader(), torch.device("cpu"), thresholds=thresholds)

    assert len(results) == 3
    assert [r["threshold"] for r in results] == list(thresholds)
    for r in results:
        assert set(r.keys()) == {"threshold", "iou", "dice", "precision", "recall", "f1"}
        for k in ("iou", "dice", "precision", "recall", "f1"):
            assert 0.0 <= r[k] <= 1.0


def test_sweep_thresholds_lower_threshold_never_decreases_recall():
    """A lower decision threshold predicts "changed" more liberally, so recall (of the true
    positive class) can only stay the same or increase as the threshold drops — a real, checkable
    monotonicity property of the metric, not an assumption about the specific model."""
    model = SiameseUNet(base_channels=4, comparison_mode="concat")
    thresholds = (0.3, 0.5, 0.7)
    results = sweep_thresholds(model, _make_loader(n=8), torch.device("cpu"), thresholds=thresholds)

    recalls = {r["threshold"]: r["recall"] for r in results}
    assert recalls[0.3] >= recalls[0.5] >= recalls[0.7]


def test_select_best_threshold_picks_highest_metric():
    results = [
        {"threshold": 0.3, "iou": 0.40, "dice": 0, "precision": 0, "recall": 0, "f1": 0},
        {"threshold": 0.5, "iou": 0.65, "dice": 0, "precision": 0, "recall": 0, "f1": 0},
        {"threshold": 0.7, "iou": 0.55, "dice": 0, "precision": 0, "recall": 0, "f1": 0},
    ]
    assert select_best_threshold(results, metric="iou") == 0.5


def test_select_best_threshold_ties_prefer_closest_to_default():
    results = [
        {"threshold": 0.3, "iou": 0.60, "dice": 0, "precision": 0, "recall": 0, "f1": 0},
        {"threshold": 0.5, "iou": 0.60, "dice": 0, "precision": 0, "recall": 0, "f1": 0},
        {"threshold": 0.7, "iou": 0.60, "dice": 0, "precision": 0, "recall": 0, "f1": 0},
    ]
    assert select_best_threshold(results, metric="iou") == 0.5


def test_select_best_threshold_supports_other_metrics():
    results = [
        {"threshold": 0.3, "iou": 0.5, "dice": 0, "precision": 0, "recall": 0.9, "f1": 0},
        {"threshold": 0.7, "iou": 0.6, "dice": 0, "precision": 0, "recall": 0.2, "f1": 0},
    ]
    assert select_best_threshold(results, metric="recall") == 0.3
