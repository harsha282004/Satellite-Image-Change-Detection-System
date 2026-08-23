import torch

from models.losses import BCEDiceLoss, DiceLoss, get_loss
from models.unet import BaselineChangeUNet, UNet


def test_unet_forward_pass_output_shape():
    model = UNet(in_channels=6, out_channels=1, base_channels=8)
    x = torch.randn(2, 6, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_baseline_change_unet_forward_pass():
    model = BaselineChangeUNet(base_channels=8)
    before = torch.randn(2, 3, 64, 64)
    after = torch.randn(2, 3, 64, 64)
    out = model(before, after)
    assert out.shape == (2, 1, 64, 64)


def test_baseline_change_unet_backward_and_optimizer_step():
    model = BaselineChangeUNet(base_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = BCEDiceLoss()

    before = torch.randn(2, 3, 32, 32)
    after = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()

    params_before = [p.clone() for p in model.parameters()]

    optimizer.zero_grad()
    logits = model(before, after)
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    changed = any(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(params_before, model.parameters())
    )
    assert changed, "optimizer step did not change any parameters"


def test_dice_loss_is_zero_for_perfect_prediction():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = (target * 2 - 1) * 20  # saturated logits -> sigmoid ~ exact target
    loss = DiceLoss()(logits, target)
    assert loss.item() < 1e-3


def test_bce_dice_loss_positive_and_finite():
    logits = torch.randn(2, 1, 8, 8)
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loss = BCEDiceLoss()(logits, target)
    assert torch.isfinite(loss) and loss.item() > 0


def test_get_loss_factory():
    assert isinstance(get_loss("bce"), torch.nn.BCEWithLogitsLoss)
    assert isinstance(get_loss("dice"), DiceLoss)
    assert isinstance(get_loss("bce_dice"), BCEDiceLoss)
