"""Phase 15: predict_probability correctness and predict_mask's consistency with it."""
import numpy as np
import torch

from models.siamese_unet import SiameseUNet
from src.inference.predict import predict_mask, predict_probability


def test_predict_probability_output_shape_dtype_and_range():
    model = SiameseUNet(base_channels=4, comparison_mode="diff")
    before = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    after = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)

    probs = predict_probability(model, before, after, image_size=32, device=torch.device("cpu"))

    assert probs.shape == (32, 32)
    assert probs.dtype == np.float32
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert not np.array_equal(probs, probs.astype(np.uint8)), "should be continuous, not binary"


def test_predict_mask_thresholds_predict_probability_consistently():
    """predict_mask must equal (predict_probability > threshold) exactly — same underlying
    computation, not two independently-computed paths that could silently diverge."""
    model = SiameseUNet(base_channels=4, comparison_mode="concat")
    before = np.random.default_rng(2).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    after = np.random.default_rng(3).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    device = torch.device("cpu")

    probs = predict_probability(model, before, after, image_size=24, device=device)
    mask = predict_mask(model, before, after, image_size=24, device=device, threshold=0.4)

    expected = (probs > 0.4).astype(np.uint8)
    assert np.array_equal(mask, expected)


def test_predict_mask_uses_same_forward_pass_as_predict_probability_across_thresholds():
    model = SiameseUNet(base_channels=4, comparison_mode="diff_concat")
    before = np.zeros((32, 32, 3), dtype=np.uint8)
    after = np.full((32, 32, 3), 255, dtype=np.uint8)
    device = torch.device("cpu")

    probs = predict_probability(model, before, after, image_size=16, device=device)
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        mask = predict_mask(model, before, after, image_size=16, device=device, threshold=threshold)
        assert np.array_equal(mask, (probs > threshold).astype(np.uint8))
