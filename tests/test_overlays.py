import numpy as np

from src.visualization.overlays import create_overlay


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
