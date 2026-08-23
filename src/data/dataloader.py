"""DataLoader construction for the LEVIR-CD dataset."""
from typing import Optional

from torch.utils.data import DataLoader

from src.data.dataset import LEVIRCDDataset


def get_dataloader(
    root: str,
    split: str,
    batch_size: int = 8,
    image_size: int = 256,
    augment: Optional[bool] = None,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Build a DataLoader for one split.

    Augmentation and shuffling default to on for "train" and off for "val"/"test", but can be
    overridden explicitly.
    """
    if augment is None:
        augment = split == "train"
    if shuffle is None:
        shuffle = split == "train"

    dataset = LEVIRCDDataset(root=root, split=split, image_size=image_size, augment=augment)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == "train"),
    )
