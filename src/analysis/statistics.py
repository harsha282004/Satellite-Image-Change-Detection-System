"""Aggregate change-mask statistics: region count, total/percent changed pixels, largest/average
region size, and (only when a pixel size is explicitly provided) physical-area conversions.
"""
import numpy as np

from src.analysis.area import pixel_count_to_area
from src.analysis.regions import extract_regions


def compute_change_statistics(binary_mask: np.ndarray, pixel_size_meters: float = None,
                               min_region_pixels: int = 1) -> dict:
    binary_mask = (binary_mask > 0).astype(np.uint8)
    total_pixels = int(binary_mask.size)
    total_changed_pixels = int(binary_mask.sum())
    percent_changed = (total_changed_pixels / total_pixels * 100.0) if total_pixels else 0.0

    regions = extract_regions(binary_mask, min_region_pixels=min_region_pixels)
    region_pixel_counts = [r["pixel_count"] for r in regions]

    stats = {
        "num_regions": len(regions),
        "total_pixels": total_pixels,
        "total_changed_pixels": total_changed_pixels,
        "percent_changed": percent_changed,
        "largest_region_pixels": max(region_pixel_counts) if region_pixel_counts else 0,
        "average_region_pixels": (sum(region_pixel_counts) / len(region_pixel_counts))
                                  if region_pixel_counts else 0.0,
        "pixel_size_meters": pixel_size_meters,
        "regions": regions,
    }

    if pixel_size_meters is not None:
        stats["total_changed_area"] = pixel_count_to_area(total_changed_pixels, pixel_size_meters)
        stats["largest_region_area"] = pixel_count_to_area(stats["largest_region_pixels"], pixel_size_meters)
        stats["average_region_area_m2"] = stats["average_region_pixels"] * (pixel_size_meters ** 2)

    return stats
