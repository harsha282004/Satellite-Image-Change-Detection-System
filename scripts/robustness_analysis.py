"""Phase 15.4: robustness testing under controlled perturbations, plus failure-case mining.

Perturbations are applied to the AFTER image only (the before image stays fixed as reference) —
this simulates date-to-date variation between the two captures (different illumination/sensor
conditions, minor misregistration), the realistic scenario for a two-date change-detection pair,
rather than perturbing both images identically (which would cancel out in a way real acquisitions
rarely do). Ground truth is never modified. IoU is measured against the same, unperturbed mask.

Run with: venv/Scripts/python.exe scripts/robustness_analysis.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.dataset import LEVIRCDDataset
from src.data.preprocessing import binarize_mask, load_mask, resize_mask
from src.evaluation.robustness import PERTURBATIONS
from src.inference.predict import Predictor
from src.visualization.overlays import create_overlay

CONFIG = "configs/siamese_attention_e100.yaml"
CHECKPOINT = "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt"
N_SAMPLES = 10


def load_selected_threshold() -> float:
    report_path = Path("outputs/metrics/threshold_optimization_report.json")
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)["selected_threshold"]
    return 0.5


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_t = np.asarray(pred_mask, dtype=bool)
    gt_t = np.asarray(gt_mask, dtype=bool)
    tp = int((pred_t & gt_t).sum())
    fp = int((pred_t & ~gt_t).sum())
    fn = int((~pred_t & gt_t).sum())
    return tp / (tp + fp + fn + 1e-7)


def main() -> int:
    threshold = load_selected_threshold()
    print(f"Using selected threshold: {threshold}")

    predictor = Predictor(CONFIG, CHECKPOINT)
    dataset = LEVIRCDDataset(root="data/raw/levir_cd", split="test", image_size=256, augment=False)
    sample_names = dataset.names[:N_SAMPLES]
    print(f"Testing on {len(sample_names)} real test images: {sample_names}")

    rows = []
    worst_case = {"degradation": -1, "name": None, "perturbation": None}

    for name in sample_names:
        before_path = str(dataset.a_dir / name)
        after_path = str(dataset.b_dir / name)

        import cv2
        before_rgb = cv2.cvtColor(cv2.imread(before_path), cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(cv2.imread(after_path), cv2.COLOR_BGR2RGB)

        gt_raw = load_mask(str(dataset.label_dir / name))
        gt_mask = resize_mask(binarize_mask(gt_raw), predictor.image_size)

        baseline_mask = predictor.predict_from_arrays(before_rgb, after_rgb, threshold=threshold)
        baseline_iou = compute_iou(baseline_mask, gt_mask)
        rows.append({"image": name, "perturbation": "none (baseline)", "iou": baseline_iou,
                     "iou_degradation": 0.0})

        for pert_name, pert_fn in PERTURBATIONS.items():
            perturbed_after = pert_fn(after_rgb)
            perturbed_mask = predictor.predict_from_arrays(before_rgb, perturbed_after, threshold=threshold)
            perturbed_iou = compute_iou(perturbed_mask, gt_mask)
            degradation = baseline_iou - perturbed_iou
            rows.append({"image": name, "perturbation": pert_name, "iou": perturbed_iou,
                         "iou_degradation": degradation})

            if degradation > worst_case["degradation"]:
                worst_case = {"degradation": degradation, "name": name, "perturbation": pert_name,
                               "before_rgb": before_rgb, "after_rgb": after_rgb,
                               "perturbed_after": perturbed_after, "baseline_mask": baseline_mask,
                               "perturbed_mask": perturbed_mask, "gt_mask": gt_mask}

        print(f"{name}: baseline IoU={baseline_iou:.4f}")

    csv_path = Path("outputs/metrics/robustness_analysis.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "perturbation", "iou", "iou_degradation"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    print("\n=== Mean IoU degradation by perturbation type (across all images) ===")
    summary = {}
    for pert_name in PERTURBATIONS:
        degradations = [r["iou_degradation"] for r in rows if r["perturbation"] == pert_name]
        mean_deg = sum(degradations) / len(degradations)
        summary[pert_name] = mean_deg
        print(f"  {pert_name}: mean IoU degradation = {mean_deg:+.4f}")

    with open(Path("outputs/metrics/robustness_summary.json"), "w") as f:
        json.dump({
            "threshold_used": threshold,
            "n_images": len(sample_names),
            "mean_degradation_by_perturbation": summary,
            "worst_case": {"image": worst_case["name"], "perturbation": worst_case["perturbation"],
                           "degradation": worst_case["degradation"]},
        }, f, indent=2)
    print("Saved: outputs/metrics/robustness_summary.json")

    if worst_case["name"] is not None:
        save_worst_case_visualization(worst_case, Path("outputs/visualizations/robustness"))

    return 0


def save_worst_case_visualization(case: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"Worst-case robustness degradation: {case['name']}, "
                 f"perturbation={case['perturbation']}, IoU drop={case['degradation']:.4f}")

    axes[0, 0].imshow(case["before_rgb"]); axes[0, 0].set_title("Before (unperturbed)"); axes[0, 0].axis("off")
    axes[0, 1].imshow(case["after_rgb"]); axes[0, 1].set_title("After (original)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(case["gt_mask"], cmap="gray"); axes[0, 2].set_title("Ground Truth"); axes[0, 2].axis("off")
    axes[0, 3].imshow(case["baseline_mask"], cmap="gray"); axes[0, 3].set_title("Prediction (original)"); axes[0, 3].axis("off")

    axes[1, 0].imshow(case["before_rgb"]); axes[1, 0].set_title("Before (unperturbed)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(case["perturbed_after"]); axes[1, 1].set_title(f"After ({case['perturbation']})"); axes[1, 1].axis("off")

    gt = case["gt_mask"].astype(bool)
    pred = case["perturbed_mask"].astype(bool)
    diff = np.zeros((*gt.shape, 3), dtype=np.float32)
    diff[pred & ~gt] = [1.0, 1.0, 0.0]  # false positive
    diff[~pred & gt] = [0.0, 0.3, 1.0]  # false negative
    diff[pred & gt] = [0.0, 1.0, 0.0]   # true positive
    axes[1, 2].imshow(diff); axes[1, 2].set_title("Diff (FP=yellow, FN=blue)"); axes[1, 2].axis("off")
    axes[1, 3].imshow(case["perturbed_mask"], cmap="gray"); axes[1, 3].set_title("Prediction (perturbed)"); axes[1, 3].axis("off")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"worst_case_{case['name']}"
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved worst-case visualization: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
