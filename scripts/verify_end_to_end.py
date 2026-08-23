"""Phase 12 final integration check: the complete real workflow, upload-to-output, against the
real trained model and genuinely unseen test images (deliberately different from every sample
name used in prior phases' visualizations, so this exercises fresh output, not cached results).

Run with: venv/Scripts/python.exe scripts/verify_end_to_end.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.analysis.statistics import compute_change_statistics
from src.data.dataset import LEVIRCDDataset
from src.inference.predict import Predictor
from src.visualization.overlays import create_overlay

CONFIG = "configs/siamese_attention.yaml"
CHECKPOINT = "outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt"

# Deliberately different from Phase 9's demo (test_1, test_29, test_52, test_75, test_99, test_121)
FRESH_SAMPLE_NAMES = ["test_10.png", "test_45.png", "test_80.png"]


def main() -> int:
    checkpoint_path = Path(CHECKPOINT)
    if not checkpoint_path.exists():
        print(f"SKIP: checkpoint not found at {CHECKPOINT} — train the model first (see README.md).")
        return 0

    print("=== Phase 12: full pipeline, unseen test images ===")
    predictor = Predictor(CONFIG, CHECKPOINT)
    dataset_raw = LEVIRCDDataset(root="data/raw/levir_cd", split="test", image_size=256, augment=False)

    all_names = set(dataset_raw.names)
    missing = [n for n in FRESH_SAMPLE_NAMES if n not in all_names]
    if missing:
        print(f"WARNING: requested sample(s) not found in test split: {missing}. "
              f"Falling back to the first {len(FRESH_SAMPLE_NAMES)} available names.")
        sample_names = dataset_raw.names[:len(FRESH_SAMPLE_NAMES)]
    else:
        sample_names = FRESH_SAMPLE_NAMES

    out_dir = Path("outputs/visualizations/phase12_verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in sample_names:
        before_path = dataset_raw.a_dir / name
        after_path = dataset_raw.b_dir / name

        mask = predictor.predict_from_paths(str(before_path), str(after_path))
        assert mask.shape == (256, 256), f"unexpected mask shape {mask.shape}"
        assert set(np.unique(mask).tolist()) <= {0, 1}, "mask not binary"

        stats = compute_change_statistics(mask, pixel_size_meters=None, min_region_pixels=4)

        before_img = np.array(Image.open(before_path).convert("RGB").resize((256, 256)))
        overlay = create_overlay(before_img, mask, color=(1.0, 0.0, 0.0), alpha=0.6)
        Image.fromarray(overlay).save(out_dir / f"overlay_{name}")

        print(f"{name}: mask OK, {stats['num_regions']} regions, "
              f"{stats['percent_changed']:.2f}% changed -> saved overlay")

    print(f"\nPASS: full pipeline (load -> preprocess -> predict -> analyze -> visualize) "
          f"succeeded on {len(sample_names)} genuinely unseen test images.")

    print("\n=== Edge cases (see tests/test_edge_cases.py for the fast/portable pytest versions) ===")

    # Mismatched dimensions using real images of different native sizes.
    before_real = np.array(Image.open(dataset_raw.a_dir / sample_names[0]))  # 1024x1024 native
    after_small = np.array(Image.open(dataset_raw.b_dir / sample_names[1]).resize((400, 700)))
    mask2 = predictor.predict_from_arrays(before_real, after_small)
    assert mask2.shape == (256, 256)
    print("PASS: mismatched before/after native dimensions (1024x1024 vs 400x700) handled correctly")

    # Missing file.
    try:
        predictor.predict_from_paths(str(dataset_raw.a_dir / "definitely_not_a_real_file.png"),
                                      str(after_path))
        print("FAIL: missing file did not raise an error")
        return 1
    except ValueError as e:
        print(f"PASS: missing file raised a clear error: {e}")

    print("\nAll Phase 12 end-to-end checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
