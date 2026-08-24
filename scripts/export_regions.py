"""Phase 16: full region-level export for real test images — connected-component region data
(geometry + prediction-probability stats) as CSV/JSON per image, plus a region-ID-labeled overlay.

`--min-region-pixels` (default 4) is an explicit, documented noise filter (Phase 16.2) — not a
hard-coded unexplained constant: at 256x256 resolution with LEVIR-CD's ~2m/pixel effective ground
sampling (src/analysis/area.py), a region smaller than 4 pixels is sub-16m^2, below what this
model's training data could reliably resolve as an intentional detection rather than prediction
noise at object boundaries.

Run with: venv/Scripts/python.exe scripts/export_regions.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.area import levir_cd_effective_pixel_size
from src.analysis.severity import compute_severity_for_regions
from src.analysis.statistics import compute_change_statistics
from src.data.dataset import LEVIRCDDataset
from src.inference.predict import Predictor
from src.visualization.overlays import create_region_id_overlay

CONFIG = "configs/siamese_attention_e100.yaml"
CHECKPOINT = "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt"
SAMPLE_NAMES = ["test_29.png", "test_45.png", "test_52.png", "test_75.png", "test_99.png"]


def region_row_for_csv(image_name: str, region: dict, pixel_size_m: float) -> dict:
    row = {
        "image": image_name,
        "region_id": region["id"],
        "pixel_count": region["pixel_count"],
        "width_px": region["width"],
        "height_px": region["height"],
        "perimeter_px": round(region["perimeter"], 2),
        "aspect_ratio": round(region["aspect_ratio"], 3),
        "change_density": round(region["change_density"], 3),
        "centroid_row": round(region["centroid"][0], 1),
        "centroid_col": round(region["centroid"][1], 1),
        "bbox_min_row": region["bbox"]["min_row"],
        "bbox_max_row": region["bbox"]["max_row"],
        "bbox_min_col": region["bbox"]["min_col"],
        "bbox_max_col": region["bbox"]["max_col"],
        "area_m2": round(region["pixel_count"] * pixel_size_m ** 2, 1),
        "mean_prediction_probability": round(region.get("mean_prediction_probability", float("nan")), 4),
        "max_prediction_probability": round(region.get("max_prediction_probability", float("nan")), 4),
        "severity_score": round(region.get("severity_score", float("nan")), 2),
        "severity_category": region.get("severity_category", ""),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-region-pixels", type=int, default=4,
                         help="Discard predicted regions smaller than this (noise filter, "
                              "documented in this script's module docstring)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/regions"))
    args = parser.parse_args()

    predictor = Predictor(CONFIG, CHECKPOINT)
    dataset = LEVIRCDDataset(root="data/raw/levir_cd", split="test", image_size=256, augment=False)
    pixel_size_m = levir_cd_effective_pixel_size(predictor.image_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_report = []

    for name in SAMPLE_NAMES:
        before_path = dataset.a_dir / name
        after_path = dataset.b_dir / name

        prob_map = predictor.predict_probability_from_paths(str(before_path), str(after_path))
        mask = (prob_map > 0.4).astype(np.uint8)  # Phase 15.2's validation-selected threshold

        stats = compute_change_statistics(
            mask, probability_map=prob_map, pixel_size_meters=pixel_size_m,
            min_region_pixels=args.min_region_pixels,
        )
        stats["regions"] = compute_severity_for_regions(stats["regions"])

        for region in stats["regions"]:
            all_rows.append(region_row_for_csv(name, region, pixel_size_m))

        all_report.append({
            "image": name,
            "num_regions": stats["num_regions"],
            "total_changed_pixels": stats["total_changed_pixels"],
            "percent_changed": stats["percent_changed"],
            "largest_region_pixels": stats["largest_region_pixels"],
            "smallest_region_pixels": stats["smallest_region_pixels"],
            "average_region_pixels": stats["average_region_pixels"],
            "regions": stats["regions"],
        })

        import cv2
        after_img = cv2.cvtColor(cv2.imread(str(after_path)), cv2.COLOR_BGR2RGB)
        after_img = cv2.resize(after_img, (predictor.image_size, predictor.image_size))
        region_overlay = create_region_id_overlay(after_img, stats["regions"])
        cv2.imwrite(str(args.out_dir / f"region_ids_{name}"), cv2.cvtColor(region_overlay, cv2.COLOR_RGB2BGR))

        print(f"{name}: {stats['num_regions']} regions, "
              f"largest={stats['largest_region_pixels']}px, smallest={stats['smallest_region_pixels']}px, "
              f"avg={stats['average_region_pixels']:.1f}px")

    csv_path = args.out_dir / "regions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {csv_path} ({len(all_rows)} regions across {len(SAMPLE_NAMES)} images)")

    json_path = args.out_dir / "regions.json"
    with open(json_path, "w") as f:
        json.dump(all_report, f, indent=2)
    print(f"Saved: {json_path}")
    print(f"Saved region-ID overlays: {args.out_dir}/region_ids_*.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
