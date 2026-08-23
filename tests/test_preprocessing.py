import numpy as np
import torch

from src.data.preprocessing import (
    binarize_mask,
    normalize_image,
    resize_image,
    resize_mask,
    to_tensor_image,
    to_tensor_mask,
)


def test_binarize_mask_thresholds_antialiased_values():
    mask = np.array([[0, 100, 156, 200, 255]], dtype=np.uint8)
    result = binarize_mask(mask, threshold=127)
    assert result.tolist() == [[0, 0, 1, 1, 1]]
    assert result.dtype == np.uint8


def test_normalize_image_scales_to_unit_range():
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    result = normalize_image(image)
    assert result.dtype == np.float32
    assert np.isclose(result[0, 0, 0], 0.0)
    assert np.isclose(result[0, 0, 2], 1.0)


def test_resize_image_and_mask_change_spatial_dims():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    assert resize_image(image, 32).shape == (32, 32, 3)
    assert resize_mask(mask, 32).shape == (32, 32)


def test_resize_mask_preserves_binary_values_with_nearest_neighbor():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    resized = resize_mask(mask, 8)
    assert set(np.unique(resized).tolist()) <= {0, 1}


def test_to_tensor_image_produces_chw_float_tensor():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    t = to_tensor_image(image)
    assert t.shape == (3, 8, 8)
    assert t.dtype == torch.float32


def test_to_tensor_mask_produces_1hw_float_tensor():
    mask = np.zeros((8, 8), dtype=np.uint8)
    t = to_tensor_mask(mask)
    assert t.shape == (1, 8, 8)
    assert t.dtype == torch.float32
