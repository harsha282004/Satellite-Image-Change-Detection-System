"""Training entry point, driven by a YAML config (configs/baseline.yaml, configs/siamese.yaml).

Run with: venv/Scripts/python.exe -m src.training.train --config configs/baseline.yaml
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from models.losses import get_loss
from models.siamese_unet import SiameseUNet
from models.unet import BaselineChangeUNet
from src.data.dataloader import get_dataloader
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    extends = config.pop("extends", None)
    if extends:
        base_path = path.parent / extends
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
        merged = {**base_config, **{k: v for k, v in config.items()}}
        for key in ("dataset", "dataloader", "training", "paths"):
            if key in base_config or key in config:
                merged[key] = {**base_config.get(key, {}), **config.get(key, {})}
        config = merged

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def build_model(config: dict) -> torch.nn.Module:
    model_type = config["model"]["type"]
    if model_type == "baseline_unet":
        return BaselineChangeUNet(base_channels=config["model"].get("base_channels", 32))
    if model_type == "siamese_unet":
        return SiameseUNet(**{k: v for k, v in config["model"].items() if k != "type"})
    raise ValueError(f"Unknown model type: {model_type!r}")


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    name = config["training"]["optimizer"].lower()
    lr = config["training"]["learning_rate"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    set_seed(config["training"]["seed"])
    device = resolve_device(config["device"])
    print(f"Config: {args.config}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")

    train_loader = get_dataloader(
        root=config["dataset"]["root"],
        split="train",
        batch_size=config["dataloader"]["batch_size"],
        image_size=config["dataset"]["image_size"],
        num_workers=config["dataloader"]["num_workers"],
    )
    val_loader = get_dataloader(
        root=config["dataset"]["root"],
        split="val",
        batch_size=config["dataloader"]["batch_size"],
        image_size=config["dataset"]["image_size"],
        num_workers=config["dataloader"]["num_workers"],
    )
    print(f"train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
    print(f"val samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")

    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['type']}, parameters: {n_params:,}")

    optimizer = build_optimizer(config, model)
    loss_fn = get_loss(config["training"]["loss"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        experiment_name=config["experiment_name"],
        config=config,
    )

    best_val_iou = trainer.fit(train_loader, val_loader, num_epochs=config["training"]["epochs"])
    print(f"\nTraining complete. Best val IoU: {best_val_iou:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
