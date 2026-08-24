"""Phase 13: early stopping, LR scheduling, and optimizer/weight-decay tests.

Uses a mocked src.training.trainer.validate so early-stopping/scheduler control flow is tested
deterministically (scripted val_iou sequences) rather than depending on real training dynamics.
"""
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.losses import BCEDiceLoss
from models.siamese_unet import SiameseUNet
from src.training.train import build_optimizer, build_scheduler
from src.training.trainer import Trainer

TEST_EXPERIMENT_NAME = "phase13_pytest_trainer_tmp"


@pytest.fixture
def cleanup_test_experiment():
    yield
    shutil.rmtree(Path("outputs/experiments") / TEST_EXPERIMENT_NAME, ignore_errors=True)
    shutil.rmtree(Path("outputs/checkpoints") / TEST_EXPERIMENT_NAME, ignore_errors=True)


def _tiny_model_and_optimizer(lr=1e-3):
    model = SiameseUNet(base_channels=4, comparison_mode="diff")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def _make_loader(n=2, size=16):
    before = torch.rand(n, 3, size, size)
    after = torch.rand(n, 3, size, size)
    mask = torch.randint(0, 2, (n, 1, size, size)).float()
    return DataLoader(TensorDataset(before, after, mask), batch_size=n)


def _metrics(iou):
    return {"iou": iou, "dice": iou, "precision": iou, "recall": iou, "f1": iou, "accuracy": iou}


def _make_trainer(model, optimizer, scheduler=None, early_stopping=None):
    return Trainer(
        model=model, optimizer=optimizer, loss_fn=BCEDiceLoss(), device=torch.device("cpu"),
        checkpoint_dir="outputs/checkpoints", experiment_name=TEST_EXPERIMENT_NAME,
        config={}, scheduler=scheduler, early_stopping=early_stopping,
    )


@patch("src.training.trainer.validate")
def test_runs_full_epochs_when_early_stopping_disabled(mock_validate, cleanup_test_experiment):
    """Regression check: with no early_stopping config (the default, matching every pre-Phase-13
    config), training must run the full epoch count even if val_iou never improves."""
    mock_validate.side_effect = [(0.5, _metrics(0.3))] * 5
    model, optimizer = _tiny_model_and_optimizer()
    trainer = _make_trainer(model, optimizer)  # early_stopping=None -> disabled

    result = trainer.fit(_make_loader(), _make_loader(), num_epochs=5)

    assert result["epochs_trained"] == 5
    assert result["early_stopped"] is False
    assert result["max_epochs"] == 5


@patch("src.training.trainer.validate")
def test_early_stopping_stops_after_patience_epochs_without_improvement(mock_validate, cleanup_test_experiment):
    # val_iou improves through epoch 3 (0.1, 0.2, 0.3), then plateaus at 0.3 for epochs 4-6.
    # patience=3 -> epochs_no_improve reaches 3 at epoch 6 -> stop after epoch 6.
    mock_validate.side_effect = [
        (0.9, _metrics(0.1)), (0.8, _metrics(0.2)), (0.7, _metrics(0.3)),
        (0.7, _metrics(0.3)), (0.7, _metrics(0.3)), (0.7, _metrics(0.3)),
    ]
    model, optimizer = _tiny_model_and_optimizer()
    trainer = _make_trainer(model, optimizer, early_stopping={"enabled": True, "monitor": "val_iou", "patience": 3})

    result = trainer.fit(_make_loader(), _make_loader(), num_epochs=20)

    assert result["early_stopped"] is True
    assert result["epochs_trained"] == 6
    assert result["best_epoch"] == 3
    assert result["best_val_iou"] == pytest.approx(0.3)


@patch("src.training.trainer.validate")
def test_early_stopping_always_retains_best_checkpoint_not_last(mock_validate, cleanup_test_experiment):
    """Best checkpoint on disk must correspond to the best epoch, not the last epoch trained."""
    mock_validate.side_effect = [
        (0.5, _metrics(0.5)), (0.3, _metrics(0.7)), (0.6, _metrics(0.4)),
        (0.6, _metrics(0.4)), (0.6, _metrics(0.4)),
    ]
    model, optimizer = _tiny_model_and_optimizer()
    trainer = _make_trainer(model, optimizer, early_stopping={"enabled": True, "monitor": "val_iou", "patience": 3})

    result = trainer.fit(_make_loader(), _make_loader(), num_epochs=20)

    assert result["best_epoch"] == 2  # the 0.7 IoU epoch
    checkpoint = torch.load(trainer.checkpoint_dir / "best.pt", weights_only=False)
    assert checkpoint["epoch"] == 2


@patch("src.training.trainer.validate")
def test_scheduler_reduces_lr_on_sustained_plateau(mock_validate, cleanup_test_experiment):
    initial_lr = 1e-3
    mock_validate.side_effect = [(0.5, _metrics(0.3))] * 10  # constant val_iou -> plateau
    model, optimizer = _tiny_model_and_optimizer(lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    trainer = _make_trainer(model, optimizer, scheduler=scheduler)

    trainer.fit(_make_loader(), _make_loader(), num_epochs=10)

    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr < initial_lr


@patch("src.training.trainer.validate")
def test_no_scheduler_keeps_lr_constant(mock_validate, cleanup_test_experiment):
    initial_lr = 1e-3
    mock_validate.side_effect = [(0.5, _metrics(0.3))] * 5
    model, optimizer = _tiny_model_and_optimizer(lr=initial_lr)
    trainer = _make_trainer(model, optimizer, scheduler=None)

    trainer.fit(_make_loader(), _make_loader(), num_epochs=5)

    assert optimizer.param_groups[0]["lr"] == initial_lr


@patch("src.training.trainer.validate")
def test_lr_is_logged_every_epoch(mock_validate, cleanup_test_experiment):
    mock_validate.side_effect = [(0.5, _metrics(0.3))] * 3
    model, optimizer = _tiny_model_and_optimizer(lr=1e-3)
    trainer = _make_trainer(model, optimizer)

    trainer.fit(_make_loader(), _make_loader(), num_epochs=3)

    assert all(row.get("lr") == 1e-3 for row in trainer.logger.rows)


def test_build_optimizer_adam_default_weight_decay_zero():
    model = SiameseUNet(base_channels=4)
    config = {"training": {"optimizer": "adam", "learning_rate": 1e-4}}
    opt = build_optimizer(config, model)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_build_optimizer_adamw_with_weight_decay():
    model = SiameseUNet(base_channels=4)
    config = {"training": {"optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 0.01}}
    opt = build_optimizer(config, model)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_build_optimizer_unknown_raises():
    model = SiameseUNet(base_channels=4)
    config = {"training": {"optimizer": "rmsprop", "learning_rate": 1e-4}}
    with pytest.raises(ValueError):
        build_optimizer(config, model)


def test_build_scheduler_absent_returns_none():
    model = SiameseUNet(base_channels=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    assert build_scheduler({"training": {}}, opt) is None


def test_build_scheduler_string_none_returns_none():
    model = SiameseUNet(base_channels=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    assert build_scheduler({"training": {"scheduler": "none"}}, opt) is None


def test_build_scheduler_reduce_on_plateau_config():
    model = SiameseUNet(base_channels=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    config = {"training": {"scheduler": {"type": "reduce_on_plateau", "factor": 0.5, "patience": 4, "min_lr": 1e-6}}}
    sched = build_scheduler(config, opt)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
