"""Phase 20: real, measured architecture comparison table — every prior CNN architecture (Phase 4
baseline, Phase 5/8 Siamese variants) plus the new Transformer-based detector (Phase 20), on equal
footing: same held-out LEVIR-CD test split, same GPU, same batch size for inference timing.

Test-set IoU/Dice/Precision/Recall/F1/Accuracy are read from each model's existing
`outputs/metrics/*_test_metrics.json` (already real, measured — not re-run here to avoid the
known GPU non-determinism, `DEVELOPMENT_LOG.md` Phase 6, changing already-reported numbers).
Parameter counts and inference time are freshly measured here, for all 6 models under one
identical procedure, since inference time was never previously measured for any model in this
project.

Training time is reported only where it was actually measured (Transformer, Phase 20, this
session); the original 5 architectures' exact Phase 8 per-model training time was not recorded
(only an approximate "~25 min each" ballpark exists for Phase 4/5 in README.md) — this gap is
reported honestly, not filled in with an invented number.

Run with: venv/Scripts/python.exe scripts/architecture_comparison.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.dataloader import get_dataloader
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model, load_config, resolve_device

MODELS = [
    {"name": "Baseline U-Net", "config": "configs/baseline.yaml",
     "checkpoint": "outputs/checkpoints/baseline_unet/best.pt",
     "metrics_file": "outputs/metrics/baseline_unet_test_metrics.json"},
    {"name": "Siamese U-Net (diff)", "config": "configs/siamese_diff.yaml",
     "checkpoint": "outputs/checkpoints/siamese_unet_diff/best.pt",
     "metrics_file": "outputs/metrics/siamese_unet_diff_test_metrics.json"},
    {"name": "Siamese U-Net (concat)", "config": "configs/siamese_concat.yaml",
     "checkpoint": "outputs/checkpoints/siamese_unet_concat/best.pt",
     "metrics_file": "outputs/metrics/siamese_unet_concat_test_metrics.json"},
    {"name": "Siamese U-Net (diff_concat)", "config": "configs/siamese.yaml",
     "checkpoint": "outputs/checkpoints/siamese_unet_diff_concat/best.pt",
     "metrics_file": "outputs/metrics/siamese_unet_diff_concat_test_metrics.json"},
    {"name": "Siamese U-Net + Attention (diff_concat)", "config": "configs/siamese_attention.yaml",
     "checkpoint": "outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt",
     "metrics_file": "outputs/metrics/siamese_unet_diff_concat_attention_test_metrics.json"},
    {"name": "Transformer (diff_concat)", "config": "configs/transformer.yaml",
     "checkpoint": "outputs/checkpoints/transformer_change_diff_concat/best.pt",
     "metrics_file": "outputs/metrics/transformer_change_diff_concat_test_metrics.json"},
]

N_WARMUP = 5
N_TIMED = 50


@torch.no_grad()
def measure_inference_time_ms(model, sample_before, sample_after, device) -> float:
    """Mean single-pair (batch=1) forward-pass latency in milliseconds, GPU-synchronized timing,
    after a warmup period (first CUDA calls include one-time kernel compilation/allocation cost
    that would otherwise unfairly penalize whichever model happens to run first)."""
    model.eval()
    before = sample_before.unsqueeze(0).to(device)
    after = sample_after.unsqueeze(0).to(device)

    for _ in range(N_WARMUP):
        model(before, after)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        model(before, after)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (elapsed / N_TIMED) * 1000.0


def main() -> int:
    results = []
    test_loader = None

    for spec in MODELS:
        config = load_config(spec["config"])
        device = resolve_device(config["device"])

        model = build_model(config).to(device)
        checkpoint = load_checkpoint(spec["checkpoint"], model, map_location=device)
        n_params = sum(p.numel() for p in model.parameters())

        if test_loader is None:
            test_loader = get_dataloader(
                root=config["dataset"]["root"], split="test", batch_size=1,
                image_size=config["dataset"]["image_size"], num_workers=0,
            )
        before, after, _ = test_loader.dataset[0]
        inference_ms = measure_inference_time_ms(model, before, after, device)

        with open(spec["metrics_file"]) as f:
            test_metrics = json.load(f)["test_metrics"]

        result = {
            "name": spec["name"],
            "parameters": n_params,
            "checkpoint_epoch": checkpoint["epoch"],
            "inference_ms_per_pair": round(inference_ms, 3),
            **{k: round(test_metrics[k], 4) for k in
               ("iou", "dice", "precision", "recall", "f1", "accuracy")},
        }
        results.append(result)
        print(f"{spec['name']}: params={n_params:,}, inference={inference_ms:.2f} ms/pair, "
              f"test IoU={test_metrics['iou']:.4f}")

    out_path = Path("outputs/metrics/architecture_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "device": str(resolve_device(load_config(MODELS[0]["config"])["device"])),
            "inference_timing_procedure": f"batch=1, {N_WARMUP} warmup + {N_TIMED} timed forward "
                                           f"passes, CUDA-synchronized, first test-set sample",
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
