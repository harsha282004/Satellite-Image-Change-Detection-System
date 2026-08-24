"""Phase 18.2/18.3: pixel-space detected regions -> real geographic polygons + area.

Only ever called on a raster with a real, verified CRS/transform (`src/geospatial/raster.py::
has_georeference`) — never invents coordinates for a plain PNG/JPEG (LEVIR-CD or an arbitrary
upload). Every function here either takes an explicit `transform`/`crs` read from a real raster
file, or raises rather than guessing.
"""
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform

from src.analysis.regions import extract_regions


def region_bbox_to_polygon(region: dict, affine_transform) -> Polygon:
    """Converts one region's pixel-space bounding box (from `src/analysis/regions.py::
    extract_regions`) into a `shapely.Polygon` in the raster's native CRS, using its real affine
    transform (`rasterio.open(...).transform`) — a proper pixel->geographic mapping, not an
    approximation."""
    bbox = region["bbox"]
    corners_px = [
        (bbox["min_col"], bbox["min_row"]),
        (bbox["max_col"], bbox["min_row"]),
        (bbox["max_col"], bbox["max_row"]),
        (bbox["min_col"], bbox["max_row"]),
    ]
    corners_geo = [affine_transform * (col, row) for col, row in corners_px]
    return Polygon(corners_geo)


def polygon_area_m2(polygon: Polygon, source_crs: str) -> float:
    """Real-world area in square meters. If `source_crs` is already a projected (metric) CRS —
    true for Sentinel-2's native per-tile UTM zone — the polygon's own `.area` is already in
    square meters and is returned directly. If `source_crs` is geographic (degrees, e.g. plain
    EPSG:4326), raises: computing area directly in degree-space would be wrong, and this project
    does not silently reproject-and-guess an equal-area CRS on the caller's behalf."""
    crs = rasterio.crs.CRS.from_user_input(source_crs)
    if crs.is_geographic:
        raise ValueError(
            f"polygon_area_m2 requires a projected (metric) CRS, got geographic CRS {source_crs!r}. "
            f"Reproject to a suitable projected CRS (e.g. the source raster's native UTM zone) "
            f"before computing area — this function will not guess one for you."
        )
    return polygon.area


def polygon_to_wgs84(polygon: Polygon, source_crs: str) -> Polygon:
    """Reprojects a polygon from `source_crs` to WGS84 (EPSG:4326) — required for GeoJSON, whose
    spec mandates WGS84 lon/lat coordinates regardless of the source raster's native CRS."""
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    return shapely_transform(transformer.transform, polygon)


def regions_to_geo_features(binary_mask: np.ndarray, raster_path: str,
                             probability_map: np.ndarray = None, min_region_pixels: int = 1) -> list:
    """End-to-end: detected pixel-space regions in `binary_mask` -> a list of GeoJSON-ready
    feature dicts (WGS84 polygon geometry + region properties, including real area in m²/hectares
    computed in the raster's native projected CRS). `raster_path` must be a real georeferenced
    file (`src/geospatial/raster.py::has_georeference` — callers are expected to check this and
    fail loudly rather than call this function on a plain image)."""
    with rasterio.open(raster_path) as src:
        affine_transform = src.transform
        source_crs = src.crs

    regions = extract_regions(binary_mask, probability_map=probability_map, min_region_pixels=min_region_pixels)

    features = []
    for region in regions:
        polygon_native = region_bbox_to_polygon(region, affine_transform)
        area_m2 = polygon_area_m2(polygon_native, str(source_crs))
        polygon_wgs84 = polygon_to_wgs84(polygon_native, str(source_crs))

        properties = {
            "region_id": region["id"],
            "pixel_count": region["pixel_count"],
            "area_m2": round(area_m2, 1),
            "area_hectares": round(area_m2 / 10_000.0, 4),
            "width_px": region["width"],
            "height_px": region["height"],
            "change_density": round(region["change_density"], 3),
        }
        if "mean_prediction_probability" in region:
            properties["mean_prediction_probability"] = round(region["mean_prediction_probability"], 4)
        if "severity_score" in region:
            properties["severity_score"] = round(region["severity_score"], 2)
            properties["severity_category"] = region["severity_category"]

        features.append({
            "type": "Feature",
            "geometry": mapping(polygon_wgs84),
            "properties": properties,
        })

    return features


def features_to_geojson(features: list) -> dict:
    return {"type": "FeatureCollection", "features": features}
