import numpy as np
import pytest

from src.analysis.area import levir_cd_effective_pixel_size, pixel_count_to_area
from src.analysis.regions import extract_regions
from src.analysis.statistics import compute_change_statistics


def test_extract_regions_empty_mask_returns_no_regions():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert extract_regions(mask) == []


def test_extract_regions_single_square_region():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1  # 3 rows x 4 cols = 12 pixels
    regions = extract_regions(mask)
    assert len(regions) == 1
    r = regions[0]
    assert r["pixel_count"] == 12
    assert r["bbox"] == {"min_row": 2, "max_row": 5, "min_col": 3, "max_col": 7}
    assert r["centroid"] == pytest.approx((3.0, 4.5))


def test_extract_regions_two_separate_regions_sorted_by_size_descending():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 1     # small: 4 pixels
    mask[10:15, 10:15] = 1  # large: 25 pixels
    regions = extract_regions(mask)
    assert len(regions) == 2
    assert regions[0]["pixel_count"] == 25
    assert regions[1]["pixel_count"] == 4


def test_extract_regions_8_connectivity_merges_diagonal_touching_pixels():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1  # touches (0,0) only diagonally
    regions = extract_regions(mask)
    assert len(regions) == 1
    assert regions[0]["pixel_count"] == 2


def test_extract_regions_min_region_pixels_filters_small_regions():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, 0] = 1          # 1 pixel - should be filtered
    mask[10:15, 10:15] = 1  # 25 pixels - should survive
    regions = extract_regions(mask, min_region_pixels=5)
    assert len(regions) == 1
    assert regions[0]["pixel_count"] == 25


def test_pixel_count_to_area_correctness():
    result = pixel_count_to_area(pixel_count=100, pixel_size_meters=2.0)
    assert result["area_m2"] == pytest.approx(400.0)  # 100 px * (2.0m)^2 = 400 m^2
    assert result["area_hectares"] == pytest.approx(0.04)
    assert result["pixel_size_meters"] == 2.0


def test_levir_cd_effective_pixel_size_derivation():
    # 1024px tile at 0.5 m/px = 512m ground extent; resized to 256px -> 512/256 = 2.0 m/px
    assert levir_cd_effective_pixel_size(256) == pytest.approx(2.0)
    # resized to 512px -> 512/512 = 1.0 m/px
    assert levir_cd_effective_pixel_size(512) == pytest.approx(1.0)
    # unresized (1024px) -> back to the original 0.5 m/px
    assert levir_cd_effective_pixel_size(1024) == pytest.approx(0.5)


def test_compute_change_statistics_without_pixel_size():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # region A: 4 px
    mask[5:9, 5:9] = 1  # region B: 16 px

    stats = compute_change_statistics(mask)

    assert stats["num_regions"] == 2
    assert stats["total_pixels"] == 100
    assert stats["total_changed_pixels"] == 20
    assert stats["percent_changed"] == pytest.approx(20.0)
    assert stats["largest_region_pixels"] == 16
    assert stats["average_region_pixels"] == pytest.approx(10.0)
    assert stats["pixel_size_meters"] is None
    assert "total_changed_area" not in stats


def test_compute_change_statistics_with_pixel_size():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # 4 px

    stats = compute_change_statistics(mask, pixel_size_meters=2.0)

    assert stats["total_changed_area"]["area_m2"] == pytest.approx(16.0)  # 4px * 4m^2/px
    assert stats["largest_region_area"]["area_m2"] == pytest.approx(16.0)
    assert stats["average_region_area_m2"] == pytest.approx(16.0)


def test_compute_change_statistics_no_change_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    stats = compute_change_statistics(mask)
    assert stats["num_regions"] == 0
    assert stats["total_changed_pixels"] == 0
    assert stats["percent_changed"] == 0.0
    assert stats["largest_region_pixels"] == 0
    assert stats["average_region_pixels"] == 0.0
