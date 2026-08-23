import torch

from src.evaluation.metrics import MetricAccumulator, confusion_counts, logits_to_binary_preds


def test_logits_to_binary_preds_thresholds_at_zero_logit():
    logits = torch.tensor([-5.0, -0.1, 0.1, 5.0])
    preds = logits_to_binary_preds(logits)
    assert preds.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_confusion_counts_correctness():
    preds = torch.tensor([1, 1, 0, 0])
    targets = torch.tensor([1, 0, 0, 1])
    tp, fp, fn, tn = confusion_counts(preds, targets)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)


def test_metric_accumulator_perfect_prediction_gives_iou_and_dice_one():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :2] = 1.0
    logits = (target * 2 - 1) * 20  # saturated -> perfect prediction after sigmoid+threshold

    acc = MetricAccumulator()
    acc.update(logits, target)
    metrics = acc.compute()

    assert metrics["iou"] > 0.999
    assert metrics["dice"] > 0.999
    assert metrics["precision"] > 0.999
    assert metrics["recall"] > 0.999
    assert metrics["f1"] > 0.999
    assert metrics["accuracy"] > 0.999


def test_metric_accumulator_all_wrong_prediction_gives_zero_iou():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :2] = 1.0
    logits = ((1 - target) * 2 - 1) * 20  # inverted prediction

    acc = MetricAccumulator()
    acc.update(logits, target)
    metrics = acc.compute()

    assert metrics["iou"] < 0.001
    assert metrics["recall"] < 0.001


def test_metric_accumulator_accumulates_across_batches():
    acc = MetricAccumulator()
    target1 = torch.ones(1, 1, 2, 2)
    logits1 = torch.ones(1, 1, 2, 2) * 20  # all correct
    target2 = torch.zeros(1, 1, 2, 2)
    logits2 = torch.ones(1, 1, 2, 2) * 20  # all wrong (predicts 1, target is 0)

    acc.update(logits1, target1)
    acc.update(logits2, target2)
    metrics = acc.compute()

    assert metrics["tp"] == 4
    assert metrics["fp"] == 4
    assert metrics["fn"] == 0
