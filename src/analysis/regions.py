"""Connected-component extraction from a binary change mask.

Input: a single-channel binary (0/1) change mask, e.g. the thresholded output of a trained model
(src/evaluation/metrics.py::logits_to_binary_preds, src/inference/predict.py::predict_mask) or a
ground-truth mask.

Per PROJECT_CONTEXT.md: regions are always called "Detected Change Region" — never "Building",
"Road", "Vegetation", "Water", etc. LEVIR-CD provides only binary building-change labels, so this
project has no basis to assign a semantic category to a region (see docs/DATASET.md,
docs/LIMITATIONS.md). That distinction is reserved for a properly labeled multi-class model
(Phase 19), if one is ever added.
"""
import cv2
import numpy as np
from scipy import ndimage


def extract_regions(binary_mask: np.ndarray, probability_map: np.ndarray = None,
                     min_region_pixels: int = 1) -> list:
    """Label 8-connected regions of `binary_mask` and return one dict per region, sorted largest
    first. Each dict:
        id, pixel_count, centroid (row, col), bbox (min/max row/col, exclusive max),
        width, height (bbox extent in pixels), perimeter (pixels, via cv2 contour arc length),
        aspect_ratio (width / height), change_density (pixel_count / bbox area — how much of the
        bounding box the region actually fills; 1.0 = a solid rectangle, lower = an irregular or
        sparse shape).

    If `probability_map` (same H×W as `binary_mask`, e.g. from
    `src/inference/predict.py::predict_probability`) is provided, each region dict also gets
    `mean_prediction_probability` and `max_prediction_probability` — the model's own sigmoid
    output statistics within that region. Omitted (not set to a placeholder) when no probability
    map is given, so callers can distinguish "not computed" from "computed as some value".

    `min_region_pixels` discards regions smaller than this (e.g. to filter single-pixel prediction
    noise) — defaults to 1 (no filtering), so filtering is an explicit opt-in, not silent. Callers
    choose this value explicitly and should document their reasoning (Phase 16.2) rather than
    relying on an unexplained default.
    """
    binary_mask = (binary_mask > 0).astype(np.uint8)
    if probability_map is not None and probability_map.shape != binary_mask.shape:
        raise ValueError(
            f"probability_map shape {probability_map.shape} does not match "
            f"binary_mask shape {binary_mask.shape}"
        )

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

        min_row, max_row = int(region_slice[0].start), int(region_slice[0].stop)
        min_col, max_col = int(region_slice[1].start), int(region_slice[1].stop)
        height = max_row - min_row
        width = max_col - min_col
        bbox_area = width * height
        aspect_ratio = (width / height) if height else 0.0
        change_density = (pixel_count / bbox_area) if bbox_area else 0.0

        contours, _ = cv2.findContours(
            region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        perimeter = float(sum(cv2.arcLength(c, closed=True) for c in contours))

        region_dict = {
            "id": region_id,
            "pixel_count": pixel_count,
            "centroid": centroid,
            "bbox": {"min_row": min_row, "max_row": max_row, "min_col": min_col, "max_col": max_col},
            "width": width,
            "height": height,
            "perimeter": perimeter,
            "aspect_ratio": aspect_ratio,
            "change_density": change_density,
        }

        if probability_map is not None:
            region_probs = probability_map[region_slice][region_mask]
            region_dict["mean_prediction_probability"] = float(region_probs.mean())
            region_dict["max_prediction_probability"] = float(region_probs.max())

        regions.append(region_dict)

    regions.sort(key=lambda r: r["pixel_count"], reverse=True)
    return regions
