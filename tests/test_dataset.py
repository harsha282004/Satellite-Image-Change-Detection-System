import pytest
import torch

from src.data.dataloader import get_dataloader
from src.data.dataset import LEVIRCDDataset


def test_dataset_loads_correct_length(synthetic_dataset_root):
    ds = LEVIRCDDataset(str(synthetic_dataset_root), "train", image_size=16, augment=False)
    assert len(ds) == 4


def test_dataset_getitem_returns_correctly_shaped_tensors(synthetic_dataset_root):
    ds = LEVIRCDDataset(str(synthetic_dataset_root), "train", image_size=16, augment=False)
    before, after, mask = ds[0]

    assert before.shape == (3, 16, 16)
    assert after.shape == (3, 16, 16)
    assert mask.shape == (1, 16, 16)
    assert before.dtype == torch.float32
    assert after.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert 0.0 <= before.min() and before.max() <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_dataset_raises_on_unpaired_files(synthetic_dataset_root_unpaired):
    with pytest.raises(ValueError, match="unpaired"):
        LEVIRCDDataset(str(synthetic_dataset_root_unpaired), "train", image_size=16)


def test_dataset_raises_on_missing_split_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        LEVIRCDDataset(str(tmp_path), "nonexistent_split", image_size=16)


def test_dataloader_produces_correctly_shaped_batches(synthetic_dataset_root):
    loader = get_dataloader(
        str(synthetic_dataset_root), "train", batch_size=2, image_size=16, num_workers=0
    )
    before, after, mask = next(iter(loader))
    assert before.shape == (2, 3, 16, 16)
    assert after.shape == (2, 3, 16, 16)
    assert mask.shape == (2, 1, 16, 16)


def test_dataloader_train_defaults_to_shuffle_and_augment(synthetic_dataset_root):
    loader = get_dataloader(str(synthetic_dataset_root), "train", batch_size=2, image_size=16)
    assert loader.dataset.augment is True


def test_dataloader_val_defaults_to_no_augment(synthetic_dataset_root):
    (synthetic_dataset_root / "val").mkdir()
    for sub in ("A", "B", "label"):
        (synthetic_dataset_root / "val" / sub).mkdir()
    # Reuse train's files as a stand-in val split for this shuffle/augment-default check only.
    import shutil
    for sub in ("A", "B", "label"):
        for f in (synthetic_dataset_root / "train" / sub).iterdir():
            shutil.copy(f, synthetic_dataset_root / "val" / sub / f.name)

    loader = get_dataloader(str(synthetic_dataset_root), "val", batch_size=2, image_size=16)
    assert loader.dataset.augment is False
