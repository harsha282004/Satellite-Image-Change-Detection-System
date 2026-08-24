"""Reusable overlay rendering: predicted-change-mask-on-image compositing.

Factored out here once a third real consumer needed it (the Phase 10 dashboard, alongside
src/evaluation/evaluate.py and scripts/analyze_predictions.py's inline versions) — Rule 6 favors
avoiding duplication once genuinely shared, not abstracting speculatively ahead of need.
"""
import cv2
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


def create_region_id_overlay(base_image: np.ndarray, regions: list, box_color=(255, 0, 0),
                              text_color=(255, 255, 0)) -> np.ndarray:
    """Draws each region's bounding box and its `id` (Phase 16.3), for visually matching a region
    table row (e.g. the dashboard's) back to its location in the image.

    `base_image`: (H, W, 3) uint8 RGB. `regions`: list of dicts from
    `src/analysis/regions.py::extract_regions` (needs each region's `bbox` and `id`). Returns a
    uint8 RGB copy — the input is never modified in place.
    """
    img = base_image.copy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    img = np.ascontiguousarray(img)

    for region in regions:
        bbox = region["bbox"]
        top_left = (bbox["min_col"], bbox["min_row"])
        bottom_right = (bbox["max_col"], bbox["max_row"])
        cv2.rectangle(img, top_left, bottom_right, box_color, 1)
        cv2.putText(
            img, str(region["id"]), (bbox["min_col"], max(0, bbox["min_row"] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA,
        )

    return img
