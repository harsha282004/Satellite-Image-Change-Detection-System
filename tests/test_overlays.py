import numpy as np

from src.analysis.regions import extract_regions
from src.visualization.overlays import create_overlay, create_region_id_overlay


def test_create_overlay_uint8_full_alpha_sets_exact_color():
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1

    overlay = create_overlay(base, mask, color=(1.0, 0.0, 0.0), alpha=1.0)

    assert overlay.dtype == np.uint8
    assert tuple(overlay[0, 0]) == (255, 0, 0)
    assert tuple(overlay[1, 1]) == (0, 0, 0)  # untouched pixel unchanged


def test_create_overlay_float_input_preserved_dtype_and_range():
    base = np.zeros((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1

    overlay = create_overlay(base, mask, color=(1.0, 0.0, 0.0), alpha=1.0)

    assert overlay.dtype == np.float32
    assert np.allclose(overlay[0, 0], [1.0, 0.0, 0.0])


def test_create_overlay_partial_alpha_blends():
    base = np.zeros((2, 2, 3), dtype=np.float32)
    mask = np.ones((2, 2), dtype=np.uint8)

    overlay = create_overlay(base, mask, color=(1.0, 0.0, 0.0), alpha=0.5)

    assert np.allclose(overlay[0, 0], [0.5, 0.0, 0.0])


def test_create_overlay_no_mask_leaves_image_unchanged():
    base = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)

    overlay = create_overlay(base, mask)

    assert np.array_equal(overlay, base)


def test_create_region_id_overlay_preserves_shape_and_dtype():
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    regions = extract_regions(mask)

    overlay = create_region_id_overlay(base, regions)

    assert overlay.shape == base.shape
    assert overlay.dtype == np.uint8


def test_create_region_id_overlay_does_not_modify_input_in_place():
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    base_copy = base.copy()
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    regions = extract_regions(mask)

    create_region_id_overlay(base, regions)

    assert np.array_equal(base, base_copy)


def test_create_region_id_overlay_draws_something_for_each_region():
    base = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask[20:26, 20:26] = 1
    regions = extract_regions(mask)
    assert len(regions) == 2

    overlay = create_region_id_overlay(base, regions)

    # Boxes/text were drawn somewhere — the overlay must differ from an all-black base.
    assert overlay.sum() > 0


def test_create_region_id_overlay_empty_regions_returns_unchanged_image():
    base = np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    overlay = create_region_id_overlay(base, [])
    assert np.array_equal(overlay, base)
