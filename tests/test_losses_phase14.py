"""Phase 14: Focal+Dice, Weighted BCE+Dice, and Tversky loss correctness tests."""
import torch

from models.losses import DiceLoss, FocalDiceLoss, FocalLoss, TverskyLoss, WeightedBCEDiceLoss, get_loss


def test_focal_loss_is_finite_and_positive_on_random_input():
    logits = torch.randn(4, 1, 16, 16)
    targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
    loss = FocalLoss()(logits, targets)
    assert torch.isfinite(loss) and loss.item() > 0


def test_focal_loss_near_zero_for_confident_correct_predictions():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = (target * 2 - 1) * 20  # saturated, correct
    loss = FocalLoss()(logits, target)
    assert loss.item() < 1e-3


def test_focal_loss_alpha_weights_positive_class():
    """Higher alpha should increase the loss contribution from positive (changed) pixels
    relative to negative ones, for the same confidently-wrong prediction."""
    target_pos = torch.ones(1, 1, 4, 4)
    wrong_logits = torch.full((1, 1, 4, 4), -10.0)  # confidently predicts "no change"

    loss_low_alpha = FocalLoss(alpha=0.2, gamma=2.0)(wrong_logits, target_pos)
    loss_high_alpha = FocalLoss(alpha=0.8, gamma=2.0)(wrong_logits, target_pos)
    assert loss_high_alpha > loss_low_alpha


def test_focal_dice_loss_finite_and_gradients_flow():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loss = FocalDiceLoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_weighted_bce_dice_with_pos_weight_one_close_to_unweighted():
    """pos_weight=1.0 should make WeightedBCEDiceLoss numerically match plain BCEDiceLoss."""
    from models.losses import BCEDiceLoss

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    targets = torch.randint(0, 2, (2, 1, 8, 8)).float()

    weighted = WeightedBCEDiceLoss(pos_weight=1.0)(logits, targets)
    unweighted = BCEDiceLoss()(logits, targets)
    assert torch.isclose(weighted, unweighted, atol=1e-5)


def test_weighted_bce_dice_higher_pos_weight_penalizes_missed_positives_more():
    target_pos = torch.ones(1, 1, 4, 4)
    wrong_logits = torch.full((1, 1, 4, 4), -10.0)  # confidently predicts "no change"

    loss_low = WeightedBCEDiceLoss(pos_weight=1.0)(wrong_logits, target_pos)
    loss_high = WeightedBCEDiceLoss(pos_weight=10.0)(wrong_logits, target_pos)
    assert loss_high > loss_low


def test_tversky_loss_equals_dice_when_alpha_beta_half():
    """TI = TP/(TP + 0.5FP + 0.5FN) and Dice = 2TP/(2TP+FP+FN) are the same ratio algebraically,
    but the two classes' smoothing conventions differ (Tversky smooths TP and the denominator by
    `smooth`; this DiceLoss smooths 2*TP and the denominator by `smooth`), so they only coincide
    exactly with smooth=0 — verified here — not for an arbitrary shared smooth value."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    targets = torch.randint(0, 2, (2, 1, 8, 8)).float()

    tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=0.0)(logits, targets)
    dice = DiceLoss(smooth=0.0)(logits, targets)
    assert torch.isclose(tversky, dice, atol=1e-5)


def test_tversky_loss_zero_for_perfect_prediction():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = (target * 2 - 1) * 20
    loss = TverskyLoss()(logits, target)
    assert loss.item() < 1e-3


def test_tversky_loss_beta_weights_false_negatives_more_than_false_positives():
    """With beta > alpha (the recall-favoring default), a false negative should be penalized
    more than an equivalent-magnitude false positive."""
    # Case 1: one false negative (predict 0, target 1)
    target_fn = torch.tensor([[[[1.0]]]])
    logits_fn = torch.tensor([[[[-20.0]]]])  # predicts ~0
    loss_fn_case = TverskyLoss(alpha=0.3, beta=0.7)(logits_fn, target_fn)

    # Case 2: one false positive (predict 1, target 0)
    target_fp = torch.tensor([[[[0.0]]]])
    logits_fp = torch.tensor([[[[20.0]]]])  # predicts ~1
    loss_fp_case = TverskyLoss(alpha=0.3, beta=0.7)(logits_fp, target_fp)

    assert loss_fn_case > loss_fp_case


def test_get_loss_factory_supports_all_phase14_losses():
    assert isinstance(get_loss("focal_dice"), FocalDiceLoss)
    assert isinstance(get_loss("weighted_bce_dice"), WeightedBCEDiceLoss)
    assert isinstance(get_loss("tversky"), TverskyLoss)


def test_get_loss_factory_passes_kwargs_through():
    loss = get_loss("tversky", alpha=0.2, beta=0.8)
    assert loss.alpha == 0.2 and loss.beta == 0.8

    loss = get_loss("weighted_bce_dice", pos_weight=7.5)
    assert loss.pos_weight_value == 7.5

    loss = get_loss("focal_dice", focal_alpha=0.6, focal_gamma=1.5)
    assert loss.focal.alpha == 0.6 and loss.focal.gamma == 1.5
