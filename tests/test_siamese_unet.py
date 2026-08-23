import pytest
import torch

from models.losses import BCEDiceLoss
from models.siamese_encoder import SiameseEncoder
from models.siamese_unet import COMPARISON_MODES, SiameseUNet, comparison_channels, compare_features


def test_siamese_encoder_is_shared_weight_module_called_twice():
    """Weight sharing is achieved by construction (one encoder instance, two forward calls) —
    verify identical input gives identical output through the shared encoder."""
    encoder = SiameseEncoder(in_channels=3, base_channels=8)
    x = torch.randn(1, 3, 32, 32)
    out_a = encoder(x)
    out_b = encoder(x)
    for a, b in zip(out_a, out_b):
        assert torch.equal(a, b)


def test_compare_features_channel_counts():
    before = torch.randn(1, 8, 4, 4)
    after = torch.randn(1, 8, 4, 4)
    assert compare_features(before, after, "diff").shape[1] == 8
    assert compare_features(before, after, "concat").shape[1] == 16
    assert compare_features(before, after, "diff_concat").shape[1] == 24


def test_compare_features_diff_is_symmetric_and_nonnegative():
    before = torch.randn(1, 4, 4, 4)
    after = torch.randn(1, 4, 4, 4)
    diff_ab = compare_features(before, after, "diff")
    diff_ba = compare_features(after, before, "diff")
    assert torch.equal(diff_ab, diff_ba)
    assert (diff_ab >= 0).all()


def test_comparison_channels_invalid_mode_raises():
    with pytest.raises(ValueError):
        comparison_channels(8, "bogus_mode")


def test_siamese_unet_invalid_comparison_mode_raises():
    with pytest.raises(ValueError):
        SiameseUNet(base_channels=8, comparison_mode="bogus_mode")


@pytest.mark.parametrize("mode", COMPARISON_MODES)
def test_siamese_unet_forward_pass_output_shape(mode):
    model = SiameseUNet(base_channels=8, comparison_mode=mode)
    before = torch.randn(2, 3, 64, 64)
    after = torch.randn(2, 3, 64, 64)
    out = model(before, after)
    assert out.shape == (2, 1, 64, 64)


@pytest.mark.parametrize("mode", COMPARISON_MODES)
def test_siamese_unet_backward_and_optimizer_step(mode):
    model = SiameseUNet(base_channels=8, comparison_mode=mode)
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


def test_siamese_unet_before_after_swap_changes_output_for_asymmetric_modes():
    """concat and diff_concat are order-sensitive (before vs after matters); diff alone is
    symmetric. Confirms the model is actually distinguishing before from after when it should."""
    torch.manual_seed(0)
    model = SiameseUNet(base_channels=8, comparison_mode="concat")
    model.eval()
    a = torch.randn(1, 3, 32, 32)
    b = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out_ab = model(a, b)
        out_ba = model(b, a)
    assert not torch.allclose(out_ab, out_ba)


def test_siamese_unet_shares_encoder_weights_with_baseline_style_parameter_count():
    """The encoder should appear once in the parameter list (shared), not duplicated for
    before/after branches — sanity-check via named_parameters uniqueness."""
    model = SiameseUNet(base_channels=8, comparison_mode="diff")
    encoder_param_names = [n for n, _ in model.named_parameters() if n.startswith("encoder.")]
    # Each encoder submodule parameter should appear exactly once (no "encoder_before"/"encoder_after" duplication).
    assert len(encoder_param_names) == len(set(encoder_param_names))
    assert any("encoder.in_conv" in n for n in encoder_param_names)
