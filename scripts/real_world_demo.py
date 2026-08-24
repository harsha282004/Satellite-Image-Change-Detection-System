"""Phase 11 real-world demonstration: fetch real Sentinel-2 imagery for a chosen location and
date pair, run the trained model, and produce a documented (not fabricated) real-world result.

This is explicitly a DEMONSTRATION, not a validated evaluation — there is no ground-truth change
mask for real-world imagery, so no IoU/Dice/precision/recall can be computed here. See
docs/REAL_WORLD_DEMO.md for the full discussion of why this differs from the benchmark evaluation
(docs/EVALUATION.md) and should not be read as having the same validity.

Default location: a Pflugerville, TX suburb (within LEVIR-CD's own source region — Austin-area
Texas — chosen for thematic continuity, not because it guarantees good results). Default dates:
2019-12-06 (earliest low-cloud L2A scene available for this tile in Earth Search) and 2024-12-19
(most recent low-cloud scene at the time this was run), both winter to reduce seasonal-lighting
confound.

Run with: venv/Scripts/python.exe scripts/real_world_demo.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.analysis.statistics import compute_change_statistics
from src.geospatial.raster import fetch_visual_crop, get_item_by_id
from src.inference.predict import Predictor
from src.realworld.validation import REAL_WORLD_DISCLAIMER, validate_real_world_input
from src.visualization.overlays import create_overlay

DEFAULT_BBOX = [-97.6500, 30.4100, -97.5900, 30.4600]  # [min_lon, min_lat, max_lon, max_lat]
DEFAULT_BEFORE_ITEM = "S2A_14RPU_20191206_1_L2A"
DEFAULT_AFTER_ITEM = "S2A_14RPU_20241219_0_L2A"

SENTINEL2_PIXEL_SIZE_M = 10.0  # native resolution of the Sentinel-2 'visual' (TCI) asset
LEVIR_CD_TRAINING_PIXEL_SIZE_M = 0.5  # docs/DATASET.md


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                         metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--before-item", type=str, default=DEFAULT_BEFORE_ITEM)
    parser.add_argument("--after-item", type=str, default=DEFAULT_AFTER_ITEM)
    parser.add_argument("--config", type=str, default="configs/siamese_attention.yaml")
    parser.add_argument("--checkpoint", type=str,
                         default="outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/real_world_demo"))
    args = parser.parse_args()

    print(f"Fetching Sentinel-2 items: before={args.before_item}, after={args.after_item}")
    print(f"Bounding box (WGS84 lon/lat): {args.bbox}")

    before_item = get_item_by_id(args.before_item, args.bbox)
    after_item = get_item_by_id(args.after_item, args.bbox)
    print(f"before date: {before_item.datetime.date()}, "
          f"cloud_cover={before_item.properties.get('eo:cloud_cover'):.3f}%")
    print(f"after date:  {after_item.datetime.date()}, "
          f"cloud_cover={after_item.properties.get('eo:cloud_cover'):.3f}%")

    before = fetch_visual_crop(before_item, args.bbox)
    after = fetch_visual_crop(after_item, args.bbox)
    print(f"Fetched crops: before={before.shape}, after={after.shape} "
          f"(native resolution {SENTINEL2_PIXEL_SIZE_M} m/pixel)")

    print(f"\n*** {REAL_WORLD_DISCLAIMER} ***")
    validation_report = validate_real_world_input(before, after, pixel_size_meters=SENTINEL2_PIXEL_SIZE_M)
    resolution_gap = validation_report["resolution"]["resolution_ratio"]
    print(f"NOTE: this is {resolution_gap:.0f}x coarser than the model's LEVIR-CD training "
          f"resolution ({LEVIR_CD_TRAINING_PIXEL_SIZE_M} m/pixel) — see docs/REAL_WORLD_DEMO.md")
    if validation_report["warnings"]:
        print(f"Phase 22 input validation ({len(validation_report['warnings'])} warning(s)):")
        for w in validation_report["warnings"]:
            print(f"  - {w}")
    else:
        print("Phase 22 input validation: no warnings (dimensions/registration/cloud heuristic).")

    predictor = Predictor(args.config, args.checkpoint)
    mask = predictor.predict_from_arrays(before, after)
    changed_fraction = mask.mean()
    print(f"Predicted mask: {mask.sum()} / {mask.size} pixels changed ({changed_fraction*100:.2f}%)")

    stats = compute_change_statistics(mask, pixel_size_meters=None, min_region_pixels=4)
    print(f"Regions detected (>=4px): {stats['num_regions']}")
    print("NOTE: no ground truth exists for this real-world pair — these are UNVALIDATED "
          "predictions, not measured against any known-correct answer. See docs/REAL_WORLD_DEMO.md.")

    import cv2
    size = predictor.image_size
    before_r = cv2.resize(before, (size, size))
    after_r = cv2.resize(after, (size, size))
    overlay = create_overlay(after_r, mask, color=(1.0, 0.0, 0.0), alpha=0.6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(before_r).save(args.out_dir / "before.png")
    Image.fromarray(after_r).save(args.out_dir / "after.png")
    Image.fromarray((mask * 255)).save(args.out_dir / "predicted_mask.png")
    Image.fromarray(overlay).save(args.out_dir / "overlay.png")

    combined = np.concatenate([before_r, after_r, np.stack([mask * 255] * 3, axis=-1).astype(np.uint8), overlay], axis=1)
    Image.fromarray(combined).save(args.out_dir / "combined.png")

    report = {
        "bbox_wgs84": args.bbox,
        "before_item_id": args.before_item,
        "before_date": str(before_item.datetime.date()),
        "before_cloud_cover_pct": before_item.properties.get("eo:cloud_cover"),
        "after_item_id": args.after_item,
        "after_date": str(after_item.datetime.date()),
        "after_cloud_cover_pct": after_item.properties.get("eo:cloud_cover"),
        "sentinel2_native_pixel_size_m": SENTINEL2_PIXEL_SIZE_M,
        "levir_cd_training_pixel_size_m": LEVIR_CD_TRAINING_PIXEL_SIZE_M,
        "resolution_gap_factor": resolution_gap,
        "model_config": args.config,
        "model_checkpoint": args.checkpoint,
        "predicted_changed_pixel_fraction": float(changed_fraction),
        "predicted_num_regions": stats["num_regions"],
        "ground_truth_available": False,
        "note": "No IoU/Dice/precision/recall can be reported: there is no ground-truth change "
                "mask for this real-world image pair. This is a qualitative demonstration only.",
        "disclaimer": REAL_WORLD_DISCLAIMER,
        "phase22_input_validation": validation_report,
    }
    with open(args.out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved all outputs to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
