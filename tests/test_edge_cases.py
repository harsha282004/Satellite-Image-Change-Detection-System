"""Phase 12 edge-case verification: mismatched dimensions, missing files, invalid input, and
explicit CPU-device inference — using small synthetic models/images so these run fast and
portably, without requiring the real (gitignored, multi-GB) dataset or trained checkpoints."""
import numpy as np
import pytest
import torch

from models.siamese_unet import SiameseUNet
from src.data.preprocessing import load_image, load_mask
from src.inference.predict import predict_mask, Predictor


def test_predict_mask_handles_before_after_with_different_native_dimensions():
    """Before and after images arriving at genuinely different sizes from each other (not just
    from the model's expected size) must both resize correctly and not error."""
    model = SiameseUNet(base_channels=4, comparison_mode="diff")
    before = np.zeros((800, 600, 3), dtype=np.uint8)  # portrait
    after = np.zeros((300, 900, 3), dtype=np.uint8)   # landscape, different aspect ratio too

    mask = predict_mask(model, before, after, image_size=64, device=torch.device("cpu"))
    assert mask.shape == (64, 64)


def test_predict_mask_runs_explicitly_on_cpu():
    model = SiameseUNet(base_channels=4, comparison_mode="concat")
    before = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    after = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)

    device = torch.device("cpu")
    mask = predict_mask(model, before, after, image_size=32, device=device)
    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_predict_mask_runs_explicitly_on_gpu_and_matches_cpu_closely():
    """CPU and GPU forward passes should agree closely (not bit-exact, given documented GPU
    non-determinism — DEVELOPMENT_LOG.md Phase 6) on the same random-init model and input."""
    torch.manual_seed(0)
    model_cpu = SiameseUNet(base_channels=4, comparison_mode="diff_concat")
    model_gpu = SiameseUNet(base_channels=4, comparison_mode="diff_concat")
    model_gpu.load_state_dict(model_cpu.state_dict())
    model_gpu = model_gpu.to("cuda")

    rng = np.random.default_rng(42)
    before = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    after = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    mask_cpu = predict_mask(model_cpu, before, after, image_size=64, device=torch.device("cpu"))
    mask_gpu = predict_mask(model_gpu, before, after, image_size=64, device=torch.device("cuda"))

    agreement = (mask_cpu == mask_gpu).mean()
    assert agreement > 0.95, f"CPU/GPU predictions disagree on {100*(1-agreement):.1f}% of pixels"


def test_load_image_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="Could not read image"):
        load_image(str(missing_path))


def test_load_mask_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="Could not read mask"):
        load_mask(str(missing_path))


def test_load_image_invalid_file_raises_clear_error(tmp_path):
    fake_image = tmp_path / "not_an_image.png"
    fake_image.write_bytes(b"this is not a valid png file")
    with pytest.raises(ValueError, match="Could not read image"):
        load_image(str(fake_image))


def test_predictor_missing_config_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        Predictor("configs/does_not_exist.yaml", "outputs/checkpoints/baseline_unet/best.pt")


def test_predictor_missing_checkpoint_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        Predictor("configs/baseline.yaml", "outputs/checkpoints/does_not_exist/best.pt")
