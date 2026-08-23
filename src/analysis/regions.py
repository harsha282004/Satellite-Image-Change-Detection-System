"""Connected-component extraction from a binary change mask.

Input: a single-channel binary (0/1) change mask, e.g. the thresholded output of a trained model
(src/evaluation/metrics.py::logits_to_binary_preds) or a ground-truth mask.
"""
import numpy as np
from scipy import ndimage


def extract_regions(binary_mask: np.ndarray, min_region_pixels: int = 1) -> list:
    """Label 8-connected regions of `binary_mask` and return one dict per region, sorted largest
    first. Each dict: id, pixel_count, centroid (row, col), bbox (min/max row/col, exclusive max).

    `min_region_pixels` discards regions smaller than this (e.g. to filter single-pixel prediction
    noise) — defaults to 1 (no filtering), so filtering is an explicit opt-in, not silent.
    """
    binary_mask = (binary_mask > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = ndimage.label(binary_mask, structure=structure)

    regions = []
    if num_features == 0:
        return regions

    objects = ndimage.find_objects(labeled)
    for region_id in range(1, num_features + 1):
        region_slice = objects[region_id - 1]
        if region_slice is None:
            continue
        region_mask = labeled[region_slice] == region_id
        pixel_count = int(region_mask.sum())
        if pixel_count < min_region_pixels:
            continue

        local_ys, local_xs = np.where(region_mask)
        global_ys = local_ys + region_slice[0].start
        global_xs = local_xs + region_slice[1].start
        centroid = (float(global_ys.mean()), float(global_xs.mean()))

        regions.append({
            "id": region_id,
            "pixel_count": pixel_count,
            "centroid": centroid,
            "bbox": {
                "min_row": int(region_slice[0].start),
                "max_row": int(region_slice[0].stop),
                "min_col": int(region_slice[1].start),
                "max_col": int(region_slice[1].stop),
            },
        })

    regions.sort(key=lambda r: r["pixel_count"], reverse=True)
    return regions
