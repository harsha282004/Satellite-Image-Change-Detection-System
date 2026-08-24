"""Phase 18: real end-to-end geospatial change analysis.

Fetches a real, georeferenced Sentinel-2 before/after pair (same location/dates as Phase 11's
real-world demonstration — reused, not re-invented), runs the trained model, converts detected
regions to real geographic polygons using the raster's actual CRS/transform, computes real-world
area, and exports GeoJSON/CSV plus an interactive Folium map.

This is GEOSPATIAL analysis, explicitly distinguished from the IMAGE-SPACE analysis used
everywhere else in this project for LEVIR-CD PNGs (docs/DATASET.md images have no CRS — this
script never runs on them). It inherits every caveat from docs/REAL_WORLD_DEMO.md: the model was
trained on LEVIR-CD (0.5 m/pixel), this imagery is Sentinel-2 (10 m/pixel, 20x coarser), and no
ground truth exists for this real-world pair — predictions here are unvalidated.

Run with: venv/Scripts/python.exe scripts/geospatial_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv

import rasterio

from src.analysis.severity import compute_severity_for_regions
from src.geospatial.maps import build_region_map, save_map_html
from src.geospatial.polygons import features_to_geojson, regions_to_geo_features
from src.geospatial.raster import fetch_georeferenced_crop, get_item_by_id, has_georeference, read_raster_metadata
from src.inference.predict import Predictor

CONFIG = "configs/siamese_attention_e100.yaml"
CHECKPOINT = "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt"

# Same location/dates as docs/REAL_WORLD_DEMO.md (Phase 11) — reused for continuity, not re-chosen.
BBOX = [-97.6500, 30.4100, -97.5900, 30.4600]
AFTER_ITEM_ID = "S2A_14RPU_20241219_0_L2A"

OUT_DIR = Path("outputs/geospatial")


def main() -> int:
    print("=== Phase 18: Geospatial Change Analysis (real Sentinel-2 data) ===")
    print("CAVEAT (inherited from docs/REAL_WORLD_DEMO.md): model trained on LEVIR-CD imagery "
          "(0.5 m/pixel). Performance on this Sentinel-2 imagery (10 m/pixel) has NOT been "
          "independently validated. No ground truth exists for this pair.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    after_item = get_item_by_id(AFTER_ITEM_ID, BBOX)
    print(f"Fetching real georeferenced crop: {AFTER_ITEM_ID}, date={after_item.datetime.date()}")

    geotiff_path = fetch_georeferenced_crop(after_item, BBOX, str(OUT_DIR / "after_crop.tif"))
    print(f"Saved real GeoTIFF: {geotiff_path}")

    if not has_georeference(geotiff_path):
        print("ERROR: fetched raster is not georeferenced — refusing to proceed with geospatial "
              "analysis rather than inventing coordinates.")
        return 1

    metadata = read_raster_metadata(geotiff_path)
    print(f"Raster metadata (real, read from the file): {json.dumps(metadata, indent=2)}")

    # For a single-image demonstration of the geospatial pipeline (region -> polygon -> area ->
    # export -> map), run the model against the same image as both "before" and "after" is not
    # meaningful; instead reuse docs/REAL_WORLD_DEMO.md's before/after pair for prediction, then
    # georeference the resulting mask using the after-image's real transform (the mask is pixel-
    # aligned to the after image by construction of predict_mask).
    import cv2
    import numpy as np
    before_path = str(OUT_DIR / "before_crop.tif")
    before_item_id = "S2A_14RPU_20191206_1_L2A"
    before_item = get_item_by_id(before_item_id, BBOX)
    fetch_georeferenced_crop(before_item, BBOX, before_path)

    with rasterio.open(geotiff_path) as src:
        after_rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
    with rasterio.open(before_path) as src:
        before_rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))

    predictor = Predictor(CONFIG, CHECKPOINT)
    prob_map_small = predictor.predict_probability_from_arrays(before_rgb, after_rgb)
    print(f"Model predicted at {predictor.image_size}x{predictor.image_size} "
          f"(resized from the raster's native {metadata['width']}x{metadata['height']})")

    # Resize the prediction back to the raster's native pixel grid so region pixel coordinates
    # correctly correspond to the real georeferenced raster's own affine transform.
    prob_map = cv2.resize(prob_map_small, (metadata["width"], metadata["height"]))
    mask = (prob_map > 0.4).astype(np.uint8)  # Phase 15.2's validation-selected threshold

    features = regions_to_geo_features(mask, geotiff_path, probability_map=prob_map, min_region_pixels=2)
    print(f"\nDetected {len(features)} region(s) in real-world (WGS84) geographic coordinates.")

    if features:
        regions_for_severity = [
            {"id": f["properties"]["region_id"], "pixel_count": f["properties"]["pixel_count"],
             "change_density": f["properties"]["change_density"],
             "mean_prediction_probability": f["properties"].get("mean_prediction_probability", 0.0)}
            for f in features
        ]
        scored = compute_severity_for_regions(regions_for_severity)
        for feature, s in zip(features, scored):
            feature["properties"]["severity_score"] = round(s["severity_score"], 2)
            feature["properties"]["severity_category"] = s["severity_category"]

    geojson = features_to_geojson(features)
    geojson_path = OUT_DIR / "regions.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved: {geojson_path}")

    csv_path = OUT_DIR / "regions.csv"
    if features:
        with open(csv_path, "w", newline="") as f:
            fieldnames = list(features[0]["properties"].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for feature in features:
                writer.writerow(feature["properties"])
        print(f"Saved: {csv_path}")

    try:
        import geopandas as gpd
        if features:
            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            gpkg_path = OUT_DIR / "regions.gpkg"
            gdf.to_file(gpkg_path, driver="GPKG")
            print(f"Saved: {gpkg_path}")
    except ImportError:
        print("geopandas not available — skipping optional GeoPackage export.")

    if features:
        fmap = build_region_map(features)
        map_path = save_map_html(fmap, str(OUT_DIR / "region_map.html"))
        print(f"Saved interactive map: {map_path}")
    else:
        print("No regions detected — skipping map generation.")

    print("\n=== Summary (real, measured) ===")
    print(f"Raster: {metadata['width']}x{metadata['height']} px, "
          f"{metadata['resolution_x']:.1f}m/pixel, CRS={metadata['crs']}")
    print(f"Regions detected: {len(features)}")
    if features:
        total_area_ha = sum(f["properties"]["area_hectares"] for f in features)
        print(f"Total detected-change area: {total_area_ha:.4f} ha (real-world, computed from "
              f"the raster's actual UTM projection, not an assumed pixel size)")
    print("\nReminder: this is a real-world demonstration, not a validated evaluation — see "
          "docs/REAL_WORLD_DEMO.md and the caveat printed at the top of this run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
