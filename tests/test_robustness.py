"""Phase 15.4: perturbation function correctness."""
import numpy as np

from src.evaluation.robustness import (
    PERTURBATIONS,
    add_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    shift_image,
)


def test_adjust_brightness_identity_at_factor_one():
    image = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    assert np.array_equal(adjust_brightness(image, 1.0), image)


def test_adjust_brightness_increases_mean_pixel_value():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    brighter = adjust_brightness(image, 1.5)
    assert brighter.mean() > image.astype(np.float32).mean()


def test_adjust_brightness_clips_to_valid_range():
    image = np.full((4, 4, 3), 250, dtype=np.uint8)
    result = adjust_brightness(image, 2.0)
    assert result.max() <= 255 and result.dtype == np.uint8


def test_adjust_contrast_identity_at_factor_one():
    image = np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    result = adjust_contrast(image, 1.0)
    assert np.allclose(result, image, atol=1)  # rounding tolerance


def test_adjust_contrast_pushes_values_away_from_mean():
    image = np.array([[[50, 50, 50], [200, 200, 200]]], dtype=np.uint8)
    result = adjust_contrast(image, 2.0)
    # the darker pixel should get darker, the brighter pixel should get brighter (or clip at bounds)
    assert result[0, 0, 0] <= image[0, 0, 0]
    assert result[0, 1, 0] >= image[0, 1, 0]


def test_add_gaussian_noise_changes_image_but_preserves_shape_and_dtype():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    noisy = add_gaussian_noise(image, sigma=15.0, seed=0)
    assert noisy.shape == image.shape
    assert noisy.dtype == np.uint8
    assert not np.array_equal(noisy, image)


def test_add_gaussian_noise_is_deterministic_given_seed():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    noisy1 = add_gaussian_noise(image, sigma=10.0, seed=42)
    noisy2 = add_gaussian_noise(image, sigma=10.0, seed=42)
    assert np.array_equal(noisy1, noisy2)


def test_shift_image_preserves_shape():
    image = np.random.default_rng(2).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    shifted = shift_image(image, dx=5, dy=5)
    assert shifted.shape == image.shape


def test_shift_image_zero_shift_is_identity():
    image = np.random.default_rng(3).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    shifted = shift_image(image, dx=0, dy=0)
    assert np.array_equal(shifted, image)


def test_shift_image_actually_moves_content():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[0, 0] = [255, 255, 255]  # distinctive marker at top-left
    shifted = shift_image(image, dx=5, dy=3)
    assert tuple(shifted[3, 5]) == (255, 255, 255)  # marker moved to (dy, dx)


def test_all_registered_perturbations_run_without_error_and_preserve_shape():
    image = np.random.default_rng(4).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    for name, fn in PERTURBATIONS.items():
        result = fn(image)
        assert result.shape == image.shape, f"{name} changed image shape"
        assert result.dtype == np.uint8, f"{name} changed image dtype"
