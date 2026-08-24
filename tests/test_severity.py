"""Phase 17: severity scoring correctness — formula transparency, category boundaries, ranking."""
import pytest

from src.analysis.severity import (
    DEFAULT_CATEGORY_THRESHOLDS,
    compute_region_severity,
    compute_severity_for_regions,
    highest_severity_regions,
    severity_category,
    severity_distribution,
)


def _region(pixel_count, mean_prob=0.5, density=0.5):
    return {"id": 1, "pixel_count": pixel_count, "mean_prediction_probability": mean_prob,
            "change_density": density}


def test_severity_category_boundaries():
    assert severity_category(0.0) == "Low"
    assert severity_category(24.9) == "Low"
    assert severity_category(25.0) == "Moderate"
    assert severity_category(49.9) == "Moderate"
    assert severity_category(50.0) == "High"
    assert severity_category(74.9) == "High"
    assert severity_category(75.0) == "Very High"
    assert severity_category(100.0) == "Very High"


def test_severity_category_custom_thresholds():
    thresholds = {"Low": 0.0, "High": 60.0}
    assert severity_category(59.9, thresholds) == "Low"
    assert severity_category(60.0, thresholds) == "High"


def test_compute_region_severity_score_in_valid_range():
    region = _region(pixel_count=250, mean_prob=0.8, density=0.9)
    result = compute_region_severity(region, total_changed_pixels_in_image=1000)
    assert 0.0 <= result["severity_score"] <= 100.0
    assert result["severity_category"] in ("Low", "Moderate", "High", "Very High")


def test_compute_region_severity_maximal_inputs_gives_maximal_score():
    """A region that is large (>= area reference), maximally confident, maximally dense, and is
    the only changed region in the image should score at (or very near) 100."""
    region = _region(pixel_count=500, mean_prob=1.0, density=1.0)
    result = compute_region_severity(region, total_changed_pixels_in_image=500)
    assert result["severity_score"] == pytest.approx(100.0, abs=0.5)


def test_compute_region_severity_minimal_inputs_gives_near_zero_score():
    region = _region(pixel_count=1, mean_prob=0.0, density=0.0)
    result = compute_region_severity(region, total_changed_pixels_in_image=10000)
    assert result["severity_score"] < 5.0


def test_compute_region_severity_larger_region_scores_higher_all_else_equal():
    small = compute_region_severity(_region(50, 0.7, 0.7), total_changed_pixels_in_image=1000)
    large = compute_region_severity(_region(400, 0.7, 0.7), total_changed_pixels_in_image=1000)
    assert large["severity_score"] > small["severity_score"]


def test_compute_region_severity_area_score_capped_at_reference():
    """Beyond AREA_REFERENCE_PIXELS, more area no longer increases the area component — it's
    capped at 1.0, not unbounded."""
    r1 = compute_region_severity(_region(500, 0.5, 0.5), total_changed_pixels_in_image=10000,
                                  area_reference_pixels=500)
    r2 = compute_region_severity(_region(5000, 0.5, 0.5), total_changed_pixels_in_image=10000,
                                  area_reference_pixels=500)
    assert r1["component_scores"]["area_score"] == pytest.approx(1.0)
    assert r2["component_scores"]["area_score"] == pytest.approx(1.0)


def test_compute_region_severity_custom_weights_change_ranking():
    """With probability weighted to 1.0 (all other weights 0), a low-probability large region
    must score below a high-probability small region — the opposite of the default weighting."""
    weights = {"area": 0.0, "probability": 1.0, "density": 0.0, "relative_size": 0.0}
    big_low_prob = compute_region_severity(_region(500, mean_prob=0.1), 1000, weights=weights)
    small_high_prob = compute_region_severity(_region(10, mean_prob=0.9), 1000, weights=weights)
    assert small_high_prob["severity_score"] > big_low_prob["severity_score"]


def test_compute_severity_for_regions_does_not_mutate_input():
    regions = [_region(100), _region(200)]
    original = [dict(r) for r in regions]
    compute_severity_for_regions(regions)
    assert regions == original


def test_compute_severity_for_regions_adds_severity_fields_to_each():
    regions = [_region(100), _region(300)]
    scored = compute_severity_for_regions(regions)
    assert len(scored) == 2
    for r in scored:
        assert "severity_score" in r and "severity_category" in r
        assert "pixel_count" in r  # original fields preserved


def test_severity_distribution_counts_and_areas():
    scored = [
        {"severity_category": "Low", "pixel_count": 10},
        {"severity_category": "Low", "pixel_count": 20},
        {"severity_category": "High", "pixel_count": 100},
    ]
    dist = severity_distribution(scored)
    assert dist["region_count_by_category"] == {"Low": 2, "High": 1}
    assert dist["changed_pixels_by_category"] == {"Low": 30, "High": 100}


def test_highest_severity_regions_sorted_descending_and_limited():
    scored = [
        {"id": 1, "severity_score": 10.0},
        {"id": 2, "severity_score": 90.0},
        {"id": 3, "severity_score": 50.0},
    ]
    top = highest_severity_regions(scored, n=2)
    assert [r["id"] for r in top] == [2, 3]
