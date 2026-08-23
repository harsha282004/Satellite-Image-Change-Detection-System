"""PyTorch Dataset for LEVIR-CD-style before/after/mask change-detection samples.

Expects the on-disk layout verified in Phase 2 (see docs/DATASET.md):
    <root>/<split>/A/<name>.png       before image
    <root>/<split>/B/<name>.png       after image
    <root>/<split>/label/<name>.png   binary change mask
"""
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from src.data.augmentation import PairedAugmentor
from src.data.preprocessing import (
    binarize_mask,
    load_image,
    load_mask,
    normalize_image,
    resize_image,
    resize_mask,
    to_tensor_image,
    to_tensor_mask,
)


class LEVIRCDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        augmentor: Optional[PairedAugmentor] = None,
    ):
        self.split_dir = Path(root) / split
        self.a_dir = self.split_dir / "A"
        self.b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"
        self.image_size = image_size
        self.augment = augment
        self.augmentor = augmentor if augmentor is not None else PairedAugmentor()

        if not (self.a_dir.exists() and self.b_dir.exists() and self.label_dir.exists()):
            raise FileNotFoundError(
                f"Expected A/, B/, label/ under {self.split_dir} — dataset not found or not "
                f"extracted. See docs/DATASET.md for acquisition instructions."
            )

        a_names = {p.name for p in self.a_dir.iterdir() if p.is_file()}
        b_names = {p.name for p in self.b_dir.iterdir() if p.is_file()}
        label_names = {p.name for p in self.label_dir.iterdir() if p.is_file()}
        common = a_names & b_names & label_names
        missing = (a_names | b_names | label_names) - common
        if missing:
            raise ValueError(
                f"{len(missing)} unpaired file(s) found in {self.split_dir} "
                f"(present in some of A/B/label but not all): {sorted(missing)[:10]}..."
            )
        if not common:
            raise ValueError(f"No paired samples found in {self.split_dir}")

        self.names = sorted(common)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]

        before = load_image(str(self.a_dir / name))
        after = load_image(str(self.b_dir / name))
        mask = binarize_mask(load_mask(str(self.label_dir / name)))

        before = resize_image(before, self.image_size)
        after = resize_image(after, self.image_size)
        mask = resize_mask(mask, self.image_size)

        if self.augment:
            before, after, mask = self.augmentor(before, after, mask)

        before_t = to_tensor_image(normalize_image(before))
        after_t = to_tensor_image(normalize_image(after))
        mask_t = to_tensor_mask(mask)

        return before_t, after_t, mask_t
