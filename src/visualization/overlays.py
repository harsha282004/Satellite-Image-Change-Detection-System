"""Reusable overlay rendering: predicted-change-mask-on-image compositing.

Factored out here once a third real consumer needed it (the Phase 10 dashboard, alongside
src/evaluation/evaluate.py and scripts/analyze_predictions.py's inline versions) — Rule 6 favors
avoiding duplication once genuinely shared, not abstracting speculatively ahead of need.
"""
import numpy as np


def create_overlay(base_image: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha: float = 1.0) -> np.ndarray:
    """`base_image`: (H, W, 3) float in [0,1] or uint8 in [0,255]. `mask`: (H, W) binary {0,1}.
    Returns an image the same shape/dtype as `base_image` with `mask` pixels blended toward
    `color` (also expected in the same value range as `base_image`) by `alpha`."""
    is_uint8 = base_image.dtype == np.uint8
    img = base_image.astype(np.float32) / 255.0 if is_uint8 else base_image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    if is_uint8:
        color_arr = color_arr if color_arr.max() <= 1.0 else color_arr / 255.0

    mask_bool = mask.astype(bool)
    img[mask_bool] = (1 - alpha) * img[mask_bool] + alpha * color_arr

    return (img * 255).astype(np.uint8) if is_uint8 else img
