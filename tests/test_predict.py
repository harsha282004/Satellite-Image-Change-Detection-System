import numpy as np
import torch

from models.siamese_unet import SiameseUNet
from src.inference.predict import predict_mask


def test_predict_mask_output_shape_and_dtype():
    model = SiameseUNet(base_channels=4, comparison_mode="diff")
    before = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    after = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)

    mask = predict_mask(model, before, after, image_size=32, device=torch.device("cpu"))

    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}


def test_predict_mask_handles_input_size_different_from_model_image_size():
    """Input arrays larger than image_size must be resized down correctly, not error."""
    model = SiameseUNet(base_channels=4, comparison_mode="concat")
    before = np.zeros((128, 128, 3), dtype=np.uint8)
    after = np.zeros((128, 128, 3), dtype=np.uint8)

    mask = predict_mask(model, before, after, image_size=16, device=torch.device("cpu"))
    assert mask.shape == (16, 16)


def test_predict_mask_threshold_affects_output():
    """A very low threshold should predict >= as many positive pixels as a high threshold."""
    model = SiameseUNet(base_channels=4, comparison_mode="diff_concat")
    before = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    after = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)

    mask_low_thresh = predict_mask(model, before, after, image_size=32,
                                    device=torch.device("cpu"), threshold=0.01)
    mask_high_thresh = predict_mask(model, before, after, image_size=32,
                                     device=torch.device("cpu"), threshold=0.99)

    assert mask_low_thresh.sum() >= mask_high_thresh.sum()
