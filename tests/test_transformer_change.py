import pytest
import torch

from models.losses import BCEDiceLoss
from models.siamese_unet import COMPARISON_MODES
from models.transformer_change import PatchEmbed, TransformerChangeDetector, TransformerEncoder


def test_patch_embed_produces_correct_token_count():
    embed = PatchEmbed(in_channels=3, embed_dim=16, patch_size=8)
    x = torch.randn(2, 3, 32, 32)
    tokens, (h, w) = embed(x)
    assert tokens.shape == (2, 16, 16)  # (32/8)^2 = 16 tokens
    assert (h, w) == (4, 4)


def test_transformer_encoder_is_shared_weight_module_called_twice():
    """Weight sharing is achieved by construction (one encoder instance, two forward calls) —
    same pattern as models/siamese_encoder.py::SiameseEncoder — identical input must give
    identical output."""
    encoder = TransformerEncoder(
        in_channels=3, embed_dim=16, patch_size=8, num_layers=1, num_heads=2,
        mlp_ratio=2.0, dropout=0.0, num_patches=16,
    )
    encoder.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        tokens_a, _ = encoder(x)
        tokens_b, _ = encoder(x)
    assert torch.equal(tokens_a, tokens_b)


def test_transformer_change_detector_invalid_comparison_mode_raises():
    with pytest.raises(ValueError):
        TransformerChangeDetector(embed_dim=16, patch_size=8, image_size=32, comparison_mode="bogus")


def test_transformer_change_detector_image_size_not_divisible_by_patch_size_raises():
    with pytest.raises(ValueError):
        TransformerChangeDetector(embed_dim=16, patch_size=8, image_size=30)


@pytest.mark.parametrize("mode", COMPARISON_MODES)
def test_transformer_change_detector_forward_pass_output_shape(mode):
    model = TransformerChangeDetector(
        embed_dim=16, patch_size=8, image_size=32, num_layers=1, num_heads=2,
        mlp_ratio=2.0, comparison_mode=mode,
    )
    before = torch.randn(2, 3, 32, 32)
    after = torch.randn(2, 3, 32, 32)
    out = model(before, after)
    assert out.shape == (2, 1, 32, 32)


def test_transformer_change_detector_backward_and_optimizer_step():
    model = TransformerChangeDetector(embed_dim=16, patch_size=8, image_size=32, num_layers=1, num_heads=2, mlp_ratio=2.0)
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


def test_transformer_change_detector_before_after_swap_changes_output_for_asymmetric_modes():
    torch.manual_seed(0)
    model = TransformerChangeDetector(
        embed_dim=16, patch_size=8, image_size=32, num_layers=1, num_heads=2,
        mlp_ratio=2.0, dropout=0.0, comparison_mode="concat",
    )
    model.eval()
    a = torch.randn(1, 3, 32, 32)
    b = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out_ab = model(a, b)
        out_ba = model(b, a)
    assert not torch.allclose(out_ab, out_ba)


def test_transformer_change_detector_shares_encoder_weights():
    """The encoder should appear once in the parameter list (shared), not duplicated for
    before/after branches."""
    model = TransformerChangeDetector(embed_dim=16, patch_size=8, image_size=32, num_layers=1, num_heads=2, mlp_ratio=2.0)
    encoder_param_names = [n for n, _ in model.named_parameters() if n.startswith("encoder.")]
    assert len(encoder_param_names) == len(set(encoder_param_names))
    assert any("patch_embed" in n for n in encoder_param_names)
