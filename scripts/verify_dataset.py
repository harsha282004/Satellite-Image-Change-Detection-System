"""LEVIR-CD dataset verification for Phase 2.

Verifies, for each requested split (A = before image, B = after image, label = binary
change mask), that:
  - A/B/label are correctly paired (same filenames, no orphans, no missing files)
  - every file actually opens (corruption check)
  - image dimensions and channel counts are consistent
  - mask pixel values are binary
  - computes the changed-vs-unchanged pixel distribution
  - saves a grid of sample (before, after, mask, overlay) visualizations

Run with: venv/Scripts/python.exe scripts/verify_dataset.py --root data/raw/levir_cd --splits val
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def verify_split(root: Path, split: str) -> dict:
    split_dir = root / split
    a_dir, b_dir, label_dir = split_dir / "A", split_dir / "B", split_dir / "label"

    report = {"split": split, "errors": []}

    if not (a_dir.exists() and b_dir.exists() and label_dir.exists()):
        report["errors"].append(f"Missing expected subfolder(s) under {split_dir}")
        return report

    a_names = {p.name for p in a_dir.iterdir() if p.is_file()}
    b_names = {p.name for p in b_dir.iterdir() if p.is_file()}
    label_names = {p.name for p in label_dir.iterdir() if p.is_file()}

    common = a_names & b_names & label_names
    only_a = a_names - b_names - label_names
    only_b = b_names - a_names - label_names
    only_label = label_names - a_names - b_names

    report["count_A"] = len(a_names)
    report["count_B"] = len(b_names)
    report["count_label"] = len(label_names)
    report["paired_count"] = len(common)
    report["unpaired_A_only"] = sorted(only_a)
    report["unpaired_B_only"] = sorted(only_b)
    report["unpaired_label_only"] = sorted(only_label)

    dims = Counter()
    modes_a = Counter()
    modes_label = Counter()
    corrupted = []
    mask_value_sets = set()
    total_changed_px = 0
    total_px = 0

    for name in sorted(common):
        try:
            with Image.open(a_dir / name) as im_a:
                im_a.verify()
            with Image.open(a_dir / name) as im_a:
                a_arr = np.array(im_a.convert("RGB"))
            with Image.open(b_dir / name) as im_b:
                im_b.verify()
            with Image.open(b_dir / name) as im_b:
                b_arr = np.array(im_b.convert("RGB"))
            with Image.open(label_dir / name) as im_l:
                im_l.verify()
            with Image.open(label_dir / name) as im_l:
                mode_label = im_l.mode
                l_arr = np.array(im_l)
        except (UnidentifiedImageError, OSError) as e:
            corrupted.append((name, str(e)))
            continue

        dims[a_arr.shape[:2]] += 1
        modes_a["RGB (converted)"] += 1
        modes_label[mode_label] += 1

        uniq = tuple(sorted(np.unique(l_arr).tolist()))
        mask_value_sets.add(uniq)

        binary_mask = l_arr > 0 if l_arr.ndim == 2 else l_arr.max(axis=-1) > 0
        total_changed_px += int(binary_mask.sum())
        total_px += int(binary_mask.size)

        if a_arr.shape[:2] != b_arr.shape[:2] or a_arr.shape[:2] != l_arr.shape[:2]:
            report["errors"].append(f"{name}: A/B/label dimension mismatch")

    report["corrupted_files"] = corrupted
    report["image_dimensions_seen"] = {str(k): v for k, v in dims.items()}
    report["a_image_mode"] = dict(modes_a)
    report["label_image_modes"] = dict(modes_label)
    report["label_unique_value_sets_seen"] = [list(s) for s in sorted(mask_value_sets)]
    report["total_pixels_checked"] = total_px
    report["total_changed_pixels"] = total_changed_px
    report["changed_pixel_fraction"] = (total_changed_px / total_px) if total_px else None

    return report


def save_sample_grid(root: Path, split: str, n_samples: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_dir = root / split
    names = sorted(p.name for p in (split_dir / "A").iterdir() if p.is_file())[:n_samples]

    fig, axes = plt.subplots(len(names), 4, figsize=(12, 3 * len(names)))
    if len(names) == 1:
        axes = axes[None, :]

    for i, name in enumerate(names):
        a = np.array(Image.open(split_dir / "A" / name).convert("RGB"))
        b = np.array(Image.open(split_dir / "B" / name).convert("RGB"))
        m = np.array(Image.open(split_dir / "label" / name).convert("L"))

        overlay = a.copy()
        overlay[m > 0] = [255, 0, 0]

        axes[i, 0].imshow(a); axes[i, 0].set_title(f"{name}\nBefore (A)"); axes[i, 0].axis("off")
        axes[i, 1].imshow(b); axes[i, 1].set_title("After (B)"); axes[i, 1].axis("off")
        axes[i, 2].imshow(m, cmap="gray"); axes[i, 2].set_title("Ground truth mask"); axes[i, 2].axis("off")
        axes[i, 3].imshow(overlay); axes[i, 3].set_title("Change overlay (red)"); axes[i, 3].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/levir_cd"))
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/metrics/dataset_verification.json"))
    parser.add_argument("--out-viz-dir", type=Path, default=Path("outputs/visualizations"))
    args = parser.parse_args()

    all_reports = {}
    for split in args.splits:
        print(f"=== Verifying split: {split} ===")
        report = verify_split(args.root, split)
        all_reports[split] = report
        print(json.dumps({k: v for k, v in report.items() if k not in
                           ("unpaired_A_only", "unpaired_B_only", "unpaired_label_only")},
                          indent=2, default=str))

        if report.get("paired_count", 0) > 0:
            viz_path = args.out_viz_dir / f"dataset_samples_{split}.png"
            save_sample_grid(args.root, split, args.samples, viz_path)
            print(f"Saved sample visualization grid: {viz_path}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\nFull verification report written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
