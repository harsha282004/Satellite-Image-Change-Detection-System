"""Phase 3 verification: dataset[0] correctness, DataLoader batching, augmentation sanity check.

Run with: venv/Scripts/python.exe scripts/verify_dataloader.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data.dataloader import get_dataloader
from src.data.dataset import LEVIRCDDataset

DATA_ROOT = "data/raw/levir_cd"
IMAGE_SIZE = 256


def check_sample(before, after, mask, label: str) -> None:
    print(f"--- {label} ---")
    print(f"before: shape={tuple(before.shape)} dtype={before.dtype} "
          f"min={before.min():.4f} max={before.max():.4f}")
    print(f"after:  shape={tuple(after.shape)} dtype={after.dtype} "
          f"min={after.min():.4f} max={after.max():.4f}")
    print(f"mask:   shape={tuple(mask.shape)} dtype={mask.dtype} "
          f"unique={torch.unique(mask).tolist()}")

    assert before.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"unexpected before shape {before.shape}"
    assert after.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"unexpected after shape {after.shape}"
    assert mask.shape == (1, IMAGE_SIZE, IMAGE_SIZE), f"unexpected mask shape {mask.shape}"
    assert before.dtype == torch.float32 and after.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert 0.0 <= before.min() and before.max() <= 1.0
    assert 0.0 <= after.min() and after.max() <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}, "mask is not strictly binary"
    print("OK: shapes/dtypes/value ranges/mask binarization all correct\n")


def main() -> int:
    print("=== Phase 3 verification: dataset[0] ===")
    ds_train_noaug = LEVIRCDDataset(DATA_ROOT, "train", image_size=IMAGE_SIZE, augment=False)
    print(f"train split size: {len(ds_train_noaug)}")
    before, after, mask = ds_train_noaug[0]
    check_sample(before, after, mask, "train[0], no augmentation")

    ds_val = LEVIRCDDataset(DATA_ROOT, "val", image_size=IMAGE_SIZE, augment=False)
    print(f"val split size: {len(ds_val)}")
    before, after, mask = ds_val[0]
    check_sample(before, after, mask, "val[0]")

    ds_test = LEVIRCDDataset(DATA_ROOT, "test", image_size=IMAGE_SIZE, augment=False)
    print(f"test split size: {len(ds_test)}")
    before, after, mask = ds_test[0]
    check_sample(before, after, mask, "test[0]")

    print("=== Phase 3 verification: augmented sample ===")
    ds_train_aug = LEVIRCDDataset(DATA_ROOT, "train", image_size=IMAGE_SIZE, augment=True)
    before, after, mask = ds_train_aug[0]
    check_sample(before, after, mask, "train[0], WITH augmentation")

    print("=== Phase 3 verification: DataLoader batching ===")
    loader = get_dataloader(DATA_ROOT, "train", batch_size=4, image_size=IMAGE_SIZE, num_workers=0)
    print(f"train loader: {len(loader)} batches of batch_size=4 (drop_last=True)")
    t0 = time.time()
    n_batches_checked = 0
    for batch_idx, (b_before, b_after, b_mask) in enumerate(loader):
        assert b_before.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert b_after.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert b_mask.shape == (4, 1, IMAGE_SIZE, IMAGE_SIZE)
        n_batches_checked += 1
        if batch_idx >= 2:
            break
    dt = time.time() - t0
    print(f"OK: iterated {n_batches_checked} batches successfully in {dt:.2f}s, "
          f"shapes correct: before/after=(4,3,{IMAGE_SIZE},{IMAGE_SIZE}), "
          f"mask=(4,1,{IMAGE_SIZE},{IMAGE_SIZE})\n")

    val_loader = get_dataloader(DATA_ROOT, "val", batch_size=4, image_size=IMAGE_SIZE, num_workers=0)
    print(f"val loader: {len(val_loader)} batches (shuffle=False, drop_last=False)")
    b_before, b_after, b_mask = next(iter(val_loader))
    print(f"first val batch shapes: before={tuple(b_before.shape)}, "
          f"after={tuple(b_after.shape)}, mask={tuple(b_mask.shape)}\n")

    print("=== Phase 3 verification: augmentation visual sanity check ===")
    save_augmentation_grid(ds_train_noaug, ds_train_aug, idx=0, n_variants=5,
                            out_path=Path("outputs/visualizations/augmentation_samples.png"))

    print("=== All Phase 3 checks passed ===")
    return 0


def save_augmentation_grid(ds_noaug, ds_aug, idx: int, n_variants: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def to_display(t: torch.Tensor) -> np.ndarray:
        return t.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(n_variants + 1, 3, figsize=(9, 3 * (n_variants + 1)))

    before, after, mask = ds_noaug[idx]
    axes[0, 0].imshow(to_display(before)); axes[0, 0].set_title("Original before"); axes[0, 0].axis("off")
    axes[0, 1].imshow(to_display(after)); axes[0, 1].set_title("Original after"); axes[0, 1].axis("off")
    axes[0, 2].imshow(mask.squeeze(0).numpy(), cmap="gray"); axes[0, 2].set_title("Original mask"); axes[0, 2].axis("off")

    for i in range(1, n_variants + 1):
        before, after, mask = ds_aug[idx]
        axes[i, 0].imshow(to_display(before)); axes[i, 0].set_title(f"Augmented before #{i}"); axes[i, 0].axis("off")
        axes[i, 1].imshow(to_display(after)); axes[i, 1].set_title(f"Augmented after #{i}"); axes[i, 1].axis("off")
        axes[i, 2].imshow(mask.squeeze(0).numpy(), cmap="gray"); axes[i, 2].set_title(f"Augmented mask #{i}"); axes[i, 2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved augmentation sample grid: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
