import torch

from models.attention import AttentionGate, AttentionUp
from models.losses import BCEDiceLoss
from models.siamese_unet import SiameseUNet


def test_attention_gate_output_shape_matches_skip():
    gate = torch.randn(2, 16, 8, 8)
    skip = torch.randn(2, 24, 8, 8)
    attn = AttentionGate(gate_channels=16, skip_channels=24)
    out = attn(gate, skip)
    assert out.shape == skip.shape


def test_attention_gate_output_bounded_by_sigmoid_times_skip():
    """Output = skip * sigmoid(...), so |output| <= |skip| elementwise."""
    gate = torch.randn(1, 8, 4, 4)
    skip = torch.randn(1, 8, 4, 4)
    attn = AttentionGate(gate_channels=8, skip_channels=8)
    out = attn(gate, skip)
    assert (out.abs() <= skip.abs() + 1e-5).all()


def test_attention_up_forward_pass_output_shape():
    up = AttentionUp(in_channels=32, skip_channels=16, out_channels=16)
    x = torch.randn(2, 32, 8, 8)
    skip = torch.randn(2, 16, 16, 16)
    out = up(x, skip)
    assert out.shape == (2, 16, 16, 16)


def test_siamese_unet_with_attention_forward_pass_output_shape():
    model = SiameseUNet(base_channels=8, comparison_mode="diff_concat", use_attention=True)
    before = torch.randn(2, 3, 64, 64)
    after = torch.randn(2, 3, 64, 64)
    out = model(before, after)
    assert out.shape == (2, 1, 64, 64)


def test_siamese_unet_with_attention_backward_and_optimizer_step():
    model = SiameseUNet(base_channels=8, comparison_mode="diff_concat", use_attention=True)
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


def test_siamese_unet_use_attention_false_matches_phase5_architecture():
    """Default behavior (use_attention=False) must be unaffected by adding the option."""
    model = SiameseUNet(base_channels=8, comparison_mode="diff", use_attention=False)
    from models.unet import Up
    assert isinstance(model.up1, Up)


def test_siamese_unet_use_attention_true_uses_attention_up():
    model = SiameseUNet(base_channels=8, comparison_mode="diff", use_attention=True)
    assert isinstance(model.up1, AttentionUp)
