"""Shared pytest fixtures. Synthetic sample datasets keep unit tests fast and independent of the
(gitignored, multi-GB) real LEVIR-CD download, per DEVELOPMENT_RULES.md Rule 5 / testing guidance.
"""
import numpy as np
import pytest
from PIL import Image


def _write_split(root, split, names, size=32, corrupt_last=False):
    a_dir = root / split / "A"
    b_dir = root / split / "B"
    label_dir = root / split / "label"
    for d in (a_dir, b_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        a = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[size // 4:size // 2, size // 4:size // 2] = 255

        Image.fromarray(a).save(a_dir / name)
        Image.fromarray(b).save(b_dir / name)
        Image.fromarray(mask, mode="L").save(label_dir / name)

    if corrupt_last:
        (a_dir / names[-1]).write_bytes(b"not a real png")


@pytest.fixture
def synthetic_dataset_root(tmp_path):
    """A tiny synthetic LEVIR-CD-shaped dataset: 4 train samples, correctly paired."""
    names = [f"sample_{i}.png" for i in range(4)]
    _write_split(tmp_path, "train", names, size=32)
    return tmp_path


@pytest.fixture
def synthetic_dataset_root_unpaired(tmp_path):
    """Same as above, but with one file missing from B/ to exercise pairing-error detection."""
    names = [f"sample_{i}.png" for i in range(4)]
    _write_split(tmp_path, "train", names, size=32)
    (tmp_path / "train" / "B" / "sample_3.png").unlink()
    return tmp_path
