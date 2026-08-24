import numpy as np
import pytest

from src.realworld.validation import (
    REAL_WORLD_DISCLAIMER,
    assess_resolution_plausibility,
    check_dimensions_match,
    estimate_registration_offset,
    screen_for_cloud_cover,
    validate_real_world_input,
)


def _natural_image(seed=0, size=64):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 180, (size, size, 3), dtype=np.uint8)


def test_check_dimensions_match_true_for_equal_shapes():
    a = _natural_image(0)
    b = _natural_image(1)
    result = check_dimensions_match(a, b)
    assert result["dimensions_match"] is True
    assert result["warning"] is None


def test_check_dimensions_match_false_and_warns_for_different_shapes():
    a = _natural_image(0, size=64)
    b = _natural_image(1, size=32)
    result = check_dimensions_match(a, b)
    assert result["dimensions_match"] is False
    assert result["warning"] is not None


def test_assess_resolution_plausibility_no_warning_for_matching_resolution():
    result = assess_resolution_plausibility(0.5)
    assert result["resolution_mismatch_warning"] is False
    assert result["warning"] is None


def test_assess_resolution_plausibility_warns_for_sentinel2_like_resolution():
    """10 m/pixel (Sentinel-2, docs/REAL_WORLD_DEMO.md) is 20x coarser than LEVIR-CD's 0.5 m/pixel
    training resolution — must trigger the mismatch warning."""
    result = assess_resolution_plausibility(10.0)
    assert result["resolution_mismatch_warning"] is True
    assert result["resolution_ratio"] == pytest.approx(20.0)
    assert result["warning"] is not None


def test_estimate_registration_offset_near_zero_for_identical_images():
    img = _natural_image(0)
    result = estimate_registration_offset(img, img.copy())
    assert result["estimated_shift_magnitude_px"] < 1.0
    assert result["likely_misregistered"] is False
    assert result["warning"] is None


def test_estimate_registration_offset_detects_shifted_image():
    img = _natural_image(0, size=128)
    shifted = np.roll(img, shift=10, axis=1)  # 10px horizontal shift
    result = estimate_registration_offset(img, shifted)
    assert result["estimated_shift_magnitude_px"] >= 3.0
    assert result["likely_misregistered"] is True
    assert result["warning"] is not None


def test_estimate_registration_offset_reports_dimension_mismatch():
    a = _natural_image(0, size=64)
    b = _natural_image(1, size=32)
    result = estimate_registration_offset(a, b)
    assert result["estimated_shift_magnitude_px"] is None
    assert result["warning"] is not None


def test_screen_for_cloud_cover_flags_bright_lowsat_image():
    bright_white = np.full((32, 32, 3), 240, dtype=np.uint8)  # white, zero saturation
    result = screen_for_cloud_cover(bright_white)
    assert result["percent_bright_lowsat_pixels"] > 90.0
    assert result["heuristic_only"] is True
    assert result["warning"] is not None


def test_screen_for_cloud_cover_does_not_flag_natural_image():
    img = _natural_image(0)
    result = screen_for_cloud_cover(img)
    assert result["warning"] is None


def test_validate_real_world_input_aggregates_warnings():
    a = _natural_image(0, size=64)
    shifted = np.roll(a, shift=15, axis=0)
    report = validate_real_world_input(a, shifted, pixel_size_meters=10.0)

    assert report["disclaimer"] == REAL_WORLD_DISCLAIMER
    assert len(report["warnings"]) >= 2  # registration + resolution, at minimum
    assert report["resolution"]["resolution_mismatch_warning"] is True
    assert report["registration"]["likely_misregistered"] is True


def test_validate_real_world_input_no_warnings_for_clean_matched_pair():
    a = _natural_image(0)
    b = _natural_image(0)  # identical -> zero shift, no cloud, matching dims
    report = validate_real_world_input(a, b, pixel_size_meters=0.5)
    assert report["warnings"] == []
