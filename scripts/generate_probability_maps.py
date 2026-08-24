"""Phase 15.1: save representative probability-map / binary-mask / overlay outputs for real
test images, using the best model and the threshold selected in Phase 15.2.

Run with: venv/Scripts/python.exe scripts/generate_probability_maps.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.dataset import LEVIRCDDataset
from src.inference.predict import Predictor
from src.visualization.overlays import create_overlay

CONFIG = "configs/siamese_attention_e100.yaml"
CHECKPOINT = "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt"
SAMPLE_NAMES = ["test_29.png", "test_45.png", "test_99.png"]  # dense-change, dramatic-change, no-change


def load_selected_threshold() -> float:
    report_path = Path("outputs/metrics/threshold_optimization_report.json")
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)["selected_threshold"]
    return 0.5


def save_probability_grid(before_img, after_img, prob_map, mask, overlay, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(title)

    axes[0].imshow(before_img); axes[0].set_title("Before"); axes[0].axis("off")
    axes[1].imshow(after_img); axes[1].set_title("After"); axes[1].axis("off")

    im = axes[2].imshow(prob_map, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Prediction Probability\n(sigmoid output, NOT calibrated confidence)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(mask, cmap="gray"); axes[3].set_title("Binary Change Mask"); axes[3].axis("off")
    axes[4].imshow(overlay); axes[4].set_title("Overlay"); axes[4].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    threshold = load_selected_threshold()
    print(f"Using selected threshold: {threshold}")

    predictor = Predictor(CONFIG, CHECKPOINT)
    dataset = LEVIRCDDataset(root="data/raw/levir_cd", split="test", image_size=256, augment=False)

    out_dir = Path("outputs/visualizations/probability_maps")
    for name in SAMPLE_NAMES:
        before_path = dataset.a_dir / name
        after_path = dataset.b_dir / name

        prob_map = predictor.predict_probability_from_paths(str(before_path), str(after_path))
        mask = (prob_map > threshold).astype(np.uint8)

        import cv2
        before_img = cv2.cvtColor(cv2.imread(str(before_path)), cv2.COLOR_BGR2RGB)
        after_img = cv2.cvtColor(cv2.imread(str(after_path)), cv2.COLOR_BGR2RGB)
        before_img = cv2.resize(before_img, (256, 256))
        after_img = cv2.resize(after_img, (256, 256))
        overlay = create_overlay(after_img, mask, color=(1.0, 0.0, 0.0), alpha=0.6)

        out_path = out_dir / f"probability_{name}"
        save_probability_grid(before_img, after_img, prob_map, mask, overlay, out_path, name)
        print(f"Saved: {out_path} (mean prob={prob_map.mean():.4f}, "
              f"max prob={prob_map.max():.4f}, changed pixels={mask.sum()})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
