"""Phase 21: real end-to-end multi-temporal change analysis — extends the two-image real-world
demonstration (Phase 11/18) to an ordered sequence of more than two real Sentinel-2 acquisitions
over the same Pflugerville, TX area of interest, computing independent per-interval change
statistics/severity for each adjacent pair (T1->T2, T2->T3, ...).

**No causal or tracking claims are made** — see `src/temporal/sequence.py`'s module docstring for
the full reasoning. Each interval below is an entirely independent two-image detection; nothing
here tracks a specific physical change across more than two images.

Inherits every caveat already documented in docs/REAL_WORLD_DEMO.md and docs/EVALUATION.md's
Phase 18 section: the model was trained on LEVIR-CD (0.5 m/pixel); this imagery is Sentinel-2
(10 m/pixel, 20x coarser); no ground truth exists for any of these real-world pairs.

Run with: venv/Scripts/python.exe scripts/multitemporal_analysis.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import rasterio

from src.geospatial.raster import fetch_georeferenced_crop, read_raster_metadata, search_sentinel2_items
from src.inference.predict import Predictor
from src.temporal.sequence import build_intervals, compute_interval_record, select_temporal_sequence

CONFIG = "configs/siamese_attention_e100.yaml"
CHECKPOINT = "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt"

# Same Pflugerville, TX area of interest as Phase 11/18 — reused for continuity, not re-chosen.
BBOX = [-97.6500, 30.4100, -97.5900, 30.4600]
DATE_RANGE = "2017-01-01/2024-12-31"
MAX_CLOUD_COVER = 5.0
N_DATES = 5
THRESHOLD = 0.40  # Phase 15.2's validation-selected threshold, reused throughout this project

OUT_DIR = Path("outputs/multitemporal")


def main() -> int:
    print("=== Phase 21: Multi-Temporal Change Analysis (real Sentinel-2 data) ===")
    print("CAVEAT (inherited from docs/REAL_WORLD_DEMO.md / Phase 18): model trained on LEVIR-CD "
          "imagery (0.5 m/pixel). Performance on this Sentinel-2 imagery (10 m/pixel) has NOT been "
          "independently validated. No ground truth exists for any pair below.")
    print("NO CAUSAL/TRACKING CLAIMS: each interval below is an independent two-image detection. "
          "A region flagged in one interval is never asserted to be the same physical change as a "
          "region flagged in another interval. See src/temporal/sequence.py for the full reasoning.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSearching real Sentinel-2 items: bbox={BBOX}, range={DATE_RANGE}, "
          f"max_cloud_cover={MAX_CLOUD_COVER}%")
    items = search_sentinel2_items(BBOX, DATE_RANGE, max_cloud_cover=MAX_CLOUD_COVER, max_items=1000)
    print(f"Found {len(items)} real, real cloud-filtered candidate dates "
          f"({items[0].datetime.date()} to {items[-1].datetime.date()})")

    sequence = select_temporal_sequence(items, n_dates=N_DATES)
    print(f"\nSelected {len(sequence)} dates, spread across the real available span:")
    for item in sequence:
        print(f"  {item.datetime.date()}  ({item.id})")

    predictor = Predictor(CONFIG, CHECKPOINT)

    crop_paths = []
    for i, item in enumerate(sequence):
        crop_path = str(OUT_DIR / f"date_{i}_{item.datetime.date()}.tif")
        fetch_georeferenced_crop(item, BBOX, crop_path)
        crop_paths.append(crop_path)
        print(f"Fetched real georeferenced crop for {item.datetime.date()}: {crop_path}")

    intervals = build_intervals(sequence)
    records = []
    for (from_item, to_item), from_path, to_path in zip(intervals, crop_paths, crop_paths[1:]):
        with rasterio.open(from_path) as src:
            before_rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        with rasterio.open(to_path) as src:
            after_rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            native_meta = read_raster_metadata(to_path)

        prob_map_small = predictor.predict_probability_from_arrays(before_rgb, after_rgb)
        prob_map = cv2.resize(prob_map_small, (native_meta["width"], native_meta["height"]))
        mask = (prob_map > THRESHOLD).astype(np.uint8)

        record = compute_interval_record(
            from_item, to_item, mask, probability_map=prob_map,
            pixel_size_meters=native_meta["resolution_x"],
        )
        records.append(record)
        print(f"\nInterval {record['from_date']} -> {record['to_date']}: "
              f"{record['num_regions']} region(s), "
              f"{record.get('total_changed_area', {}).get('area_hectares', 'n/a')} ha changed "
              f"(independent detection, not a tracked trend)")

    report_path = OUT_DIR / "temporal_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "bbox": BBOX,
            "date_range_searched": DATE_RANGE,
            "max_cloud_cover": MAX_CLOUD_COVER,
            "selected_dates": [str(item.datetime.date()) for item in sequence],
            "no_causal_claims_notice": (
                "Each interval is an independent two-image detection. No tracking, trajectory, or "
                "causal claim is made across intervals. See src/temporal/sequence.py."
            ),
            "intervals": records,
        }, f, indent=2)
    print(f"\nSaved: {report_path}")

    csv_path = OUT_DIR / "temporal_report.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["from_date", "to_date", "num_regions", "total_changed_pixels", "percent_changed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Saved: {csv_path}")

    plot_path = _plot_temporal_trend(records, OUT_DIR / "temporal_change_area.png")
    print(f"Saved temporal visualization: {plot_path}")

    print("\n=== Summary (real, measured; independent per-interval detections) ===")
    for r in records:
        area_ha = r.get("total_changed_area", {}).get("area_hectares")
        area_str = f"{area_ha:.4f} ha" if area_ha is not None else "n/a (no pixel size)"
        print(f"  {r['from_date']} -> {r['to_date']}: {r['num_regions']} regions, {area_str}")
    print("\nReminder: this is a real-world demonstration, not a validated evaluation, and NOT a "
          "tracked time series of a single change — see docs/REAL_WORLD_DEMO.md and "
          "src/temporal/sequence.py's no-causal-claims notice.")

    return 0


def _plot_temporal_trend(records: list, out_path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['from_date']}\n->{r['to_date']}" for r in records]
    areas_ha = [r.get("total_changed_area", {}).get("area_hectares", 0.0) for r in records]
    region_counts = [r["num_regions"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(len(labels)), areas_ha, color="tomato")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Detected-change area (hectares)")
    ax1.set_title("Per-interval detected change area\n(independent detections, not a tracked trend)")

    ax2.bar(range(len(labels)), region_counts, color="steelblue")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Detected region count")
    ax2.set_title("Per-interval detected region count\n(independent detections, not a tracked trend)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
