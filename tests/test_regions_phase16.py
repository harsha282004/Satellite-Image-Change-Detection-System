"""Phase 16: region-level fields (width/height/perimeter/aspect_ratio/change_density,
prediction-probability stats) and statistics.py's smallest_region_pixels."""
import numpy as np
import pytest

from src.analysis.regions import extract_regions
from src.analysis.statistics import compute_change_statistics


def test_extract_regions_solid_rectangle_geometry():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 3:9] = 1  # 4 rows x 6 cols, fully solid -> 24 pixels, bbox area 24
    regions = extract_regions(mask)

    assert len(regions) == 1
    r = regions[0]
    assert r["width"] == 6
    assert r["height"] == 4
    assert r["aspect_ratio"] == pytest.approx(6 / 4)
    assert r["change_density"] == pytest.approx(1.0)  # solid rectangle fills its own bbox exactly
    # cv2.arcLength measures along contour points at pixel centers, not the outer pixel boundary,
    # so a solid w x h rectangle's perimeter is 2*((w-1)+(h-1)), not the "outer edge" 2*(w+h).
    assert r["perimeter"] == pytest.approx(2 * ((6 - 1) + (4 - 1)))


def test_extract_regions_sparse_shape_has_lower_change_density():
    """An L-shape fills less than its full bounding box, unlike a solid rectangle."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:4, 0:2] = 1  # vertical bar
    mask[2:4, 0:4] = 1  # horizontal bar (L-shape), still 8-connected as one region
    regions = extract_regions(mask)

    assert len(regions) == 1
    assert regions[0]["change_density"] < 1.0


def test_extract_regions_without_probability_map_omits_probability_keys():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    regions = extract_regions(mask)

    assert "mean_prediction_probability" not in regions[0]
    assert "max_prediction_probability" not in regions[0]


def test_extract_regions_with_probability_map_computes_mean_and_max():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1  # 3x3 = 9 pixels

    probs = np.zeros((10, 10), dtype=np.float32)
    probs[2:5, 2:5] = np.array([
        [0.6, 0.7, 0.8],
        [0.6, 0.7, 0.9],
        [0.6, 0.7, 1.0],
    ])

    regions = extract_regions(mask, probability_map=probs)
    r = regions[0]
    expected_mean = probs[2:5, 2:5].mean()
    assert r["mean_prediction_probability"] == pytest.approx(expected_mean)
    assert r["max_prediction_probability"] == pytest.approx(1.0)


def test_extract_regions_probability_map_shape_mismatch_raises():
    mask = np.zeros((10, 10), dtype=np.uint8)
    probs = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        extract_regions(mask, probability_map=probs)


def test_compute_change_statistics_smallest_region_pixels():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 1       # 4 px (smallest)
    mask[10:15, 10:15] = 1   # 25 px (largest)
    mask[5:8, 5:8] = 1       # 9 px (middle)

    stats = compute_change_statistics(mask)

    assert stats["smallest_region_pixels"] == 4
    assert stats["largest_region_pixels"] == 25


def test_compute_change_statistics_smallest_region_pixels_zero_when_no_regions():
    mask = np.zeros((10, 10), dtype=np.uint8)
    stats = compute_change_statistics(mask)
    assert stats["smallest_region_pixels"] == 0


def test_compute_change_statistics_smallest_region_area_when_pixel_size_given():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # 4 px
    mask[10:15, 10:15] = 1  # 25 px

    stats = compute_change_statistics(mask, pixel_size_meters=2.0)
    assert stats["smallest_region_area"]["area_m2"] == pytest.approx(4 * 4.0)  # 4px * (2m)^2


def test_compute_change_statistics_passes_probability_map_through_to_regions():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    probs = np.full((10, 10), 0.75, dtype=np.float32)

    stats = compute_change_statistics(mask, probability_map=probs)
    assert stats["regions"][0]["mean_prediction_probability"] == pytest.approx(0.75)
