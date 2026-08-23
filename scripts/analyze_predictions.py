"""Phase 9 demonstration: run the best trained model on real test images, extract change regions,
compute statistics (pixel counts + physical area), and visualize the results.

Run with:
  venv/Scripts/python.exe scripts/analyze_predictions.py \
      --config configs/siamese_attention.yaml \
      --checkpoint outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.area import levir_cd_effective_pixel_size
from src.analysis.statistics import compute_change_statistics
from src.data.dataloader import get_dataloader
from src.inference.predict import Predictor
from src.training.train import load_config


def save_region_visualization(before_img, after_img, mask, regions, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    axes[0].imshow(before_img); axes[0].set_title("Before"); axes[0].axis("off")
    axes[1].imshow(after_img); axes[1].set_title("After"); axes[1].axis("off")
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title(f"Change mask + regions ({len(regions)})")
    axes[2].axis("off")

    for region in regions:
        bbox = region["bbox"]
        rect = patches.Rectangle(
            (bbox["min_col"], bbox["min_row"]),
            bbox["max_col"] - bbox["min_col"],
            bbox["max_row"] - bbox["min_row"],
            linewidth=1.2, edgecolor="red", facecolor="none",
        )
        axes[2].add_patch(rect)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--min-region-pixels", type=int, default=4,
                         help="Discard predicted regions smaller than this (noise filter)")
    args = parser.parse_args()

    config = load_config(args.config)
    predictor = Predictor(args.config, args.checkpoint)
    pixel_size_m = levir_cd_effective_pixel_size(config["dataset"]["image_size"])
    print(f"Model: {config['experiment_name']}, image_size={config['dataset']['image_size']}, "
          f"effective pixel size={pixel_size_m:.3f} m (derivation: docs/DATASET.md + "
          f"src/analysis/area.py::levir_cd_effective_pixel_size)")

    test_loader = get_dataloader(
        root=config["dataset"]["root"], split="test", batch_size=1,
        image_size=config["dataset"]["image_size"], num_workers=0,
    )
    dataset = test_loader.dataset
    indices = np.linspace(0, len(dataset) - 1, args.n_samples, dtype=int)

    all_reports = []
    for idx in indices:
        name = dataset.names[idx]
        before_t, after_t, _ = dataset[idx]
        before_img = before_t.permute(1, 2, 0).numpy()
        after_img = after_t.permute(1, 2, 0).numpy()

        # Re-run prediction from the raw (non-normalized) resized arrays via the Predictor, so
        # this script exercises the same code path Phase 10's dashboard will use.
        before_raw = (before_img * 255).astype(np.uint8)
        after_raw = (after_img * 255).astype(np.uint8)
        mask = predictor.predict_from_arrays(before_raw, after_raw)

        stats = compute_change_statistics(mask, pixel_size_meters=pixel_size_m,
                                           min_region_pixels=args.min_region_pixels)

        print(f"\n=== {name} ===")
        print(f"  regions: {stats['num_regions']}, changed pixels: {stats['total_changed_pixels']} "
              f"({stats['percent_changed']:.2f}% of tile)")
        print(f"  largest region: {stats['largest_region_pixels']} px "
              f"({stats['largest_region_area']['area_m2']:.1f} m^2)")
        print(f"  average region: {stats['average_region_pixels']:.1f} px "
              f"({stats['average_region_area_m2']:.1f} m^2)")
        print(f"  total changed area: {stats['total_changed_area']['area_m2']:.1f} m^2 "
              f"({stats['total_changed_area']['area_hectares']:.4f} ha)")

        viz_path = Path("outputs/visualizations/region_analysis") / f"{name}"
        save_region_visualization(before_img, after_img, mask, stats["regions"], viz_path, name)

        report_entry = {k: v for k, v in stats.items() if k != "regions"}
        report_entry["sample_name"] = name
        report_entry["num_regions_shown_in_viz"] = len(stats["regions"])
        all_reports.append(report_entry)

    out_json = Path("outputs/metrics/region_analysis_demo.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nSaved full report: {out_json}")
    print(f"Saved per-sample visualizations under: outputs/visualizations/region_analysis/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
