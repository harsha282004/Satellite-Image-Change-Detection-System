# DEVELOPMENT_LOG.md

Running log of phase completions. Newest entry at the top. See `PROJECT_CONTEXT.md` for phase
definitions and `DEVELOPMENT_RULES.md` for the verification rules each entry must satisfy.

---

## PHASE 4 — Baseline U-Net

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `models/unet.py`: standard 4-stage U-Net (`DoubleConv`/`Down`/`Up`, skip connections,
  `base_channels=32` for VRAM tractability) plus `BaselineChangeUNet`, which concatenates the
  before/after images into a fused 6-channel input (the Phase 4 "single fused input" baseline
  design, deliberately simpler than the Phase 5 Siamese architecture).
- `models/losses.py`: `DiceLoss`, `BCEDiceLoss` (used for the baseline), `get_loss()` factory;
  all operate on raw logits for numerical stability.
- `src/evaluation/metrics.py`: `MetricAccumulator` — accumulates TP/FP/FN/TN across a whole
  split (not per-batch averaging, which would be wrong under class imbalance) and computes
  IoU/Dice/Precision/Recall/F1/Accuracy.
- `src/training/`: `checkpoint.py` (save/load with epoch/metrics/config bundled in),
  `logger.py` (per-epoch CSV+JSON history), `validate.py` (shared val/test evaluation loop),
  `trainer.py` (`Trainer` class: train-one-epoch, fit loop, best-checkpoint-by-val-IoU
  selection — written model-agnostically so it's reused unchanged for Phase 5's Siamese model),
  `train.py` (YAML-config-driven entry point with `extends:` base-config merging, seeding, device
  auto-detection).
- `configs/config.yaml` (shared defaults) and `configs/baseline.yaml` (baseline experiment).
- `src/evaluation/evaluate.py`: test-set evaluation (real IoU/Dice/Precision/Recall/F1/Accuracy)
  plus a 6-column qualitative prediction grid (before/after/GT/prediction/overlay/diff with
  FP=yellow, FN=blue, TP=green coding).
- `docs/ARCHITECTURE.md`: documents the implemented baseline (explicitly marks the Siamese
  architecture as "Planned, not yet implemented" rather than describing it as built).
- 11 new pytest tests (`tests/test_model.py`, `tests/test_metrics.py`): forward pass and output
  shape, backward pass + optimizer step actually changes parameters, Dice loss correctness on a
  synthetic perfect prediction, metric correctness on synthetic perfect/wrong/accumulated cases.

**FILES CREATED:**
- `models/__init__.py`, `models/unet.py`, `models/losses.py`
- `src/evaluation/__init__.py`, `src/evaluation/metrics.py`, `src/evaluation/evaluate.py`
- `src/training/__init__.py`, `src/training/checkpoint.py`, `src/training/logger.py`,
  `src/training/validate.py`, `src/training/trainer.py`, `src/training/train.py`
- `configs/config.yaml`, `configs/baseline.yaml`
- `docs/ARCHITECTURE.md`
- `tests/test_model.py`, `tests/test_metrics.py`
- `outputs/checkpoints/baseline_unet/{best,last}.pt` (gitignored)
- `outputs/experiments/baseline_unet/{history.csv,history.json}` (gitignored)
- `outputs/metrics/baseline_unet_test_metrics.json` (gitignored)
- `outputs/visualizations/baseline_unet_test_predictions.png` (gitignored)

**FILES MODIFIED:**
- None outside the above

**COMMANDS EXECUTED:**
- `pytest tests/ -v` (27 tests, before training)
- `python -m src.training.train --config configs/baseline.yaml --epochs 2` (smoke test)
- `python -m src.training.train --config configs/baseline.yaml --epochs 1` (timing check: ~50s/epoch)
- `python -m src.training.train --config configs/baseline.yaml` (real 30-epoch run, GPU, background)
- `python -m src.evaluation.evaluate --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_unet/best.pt`

**TESTS:**
- `pytest tests/`: 27/27 passed (16 from Phase 3 + 11 new: `UNet`/`BaselineChangeUNet` forward
  shape, backward+optimizer-step parameter-change check, `DiceLoss`/`BCEDiceLoss` correctness,
  `get_loss` factory, `MetricAccumulator`/`confusion_counts`/`logits_to_binary_preds` correctness
  on synthetic perfect/inverted/multi-batch cases).
- 2-epoch smoke test of the full `train.py` pipeline (config loading, dataloaders, model,
  optimizer, checkpointing, CSV/JSON logging) before committing to a full run — confirmed working
  end-to-end.
- Full 30-epoch training run on the real LEVIR-CD train/val split (GPU, RTX 4050) — training loss
  and validation IoU both improved monotonically-ish across training (see history.csv), converging
  without divergence or NaN losses.
- Real, measured test-set evaluation (`src/evaluation/evaluate.py`) on the held-out LEVIR-CD test
  split (128 samples, never seen during training/validation) using the best checkpoint (selected by
  validation IoU, not by test performance — no test-set leakage into model selection).
- Manually inspected the qualitative prediction grid (6 test samples: before/after/GT/prediction/
  overlay/diff) — predictions visually track ground truth closely (majority true-positive/green in
  the diff panel), including correctly near-empty predictions on genuinely no-change scenes.

**RESULTS (actual, measured — full report: `outputs/metrics/baseline_unet_test_metrics.json`):**
```
Model: BaselineChangeUNet (base_channels=32), 7,763,905 parameters
Training: 30 epochs, Adam lr=1e-4, BCE+Dice loss, batch_size=8, image_size=256, seed=42
Best checkpoint: epoch 29 (selected by validation IoU)

Validation (at best checkpoint, epoch 29):
  IoU=0.6103  Dice=0.7580  Precision=0.7454  Recall=0.7710  Accuracy=0.9793

TEST SET (held out, 128 samples, real measured results):
  IoU=0.6250  Dice=0.7692  Precision=0.7681  Recall=0.7703  F1=0.7692  Accuracy=0.9765
```
Training time: ~50 seconds/epoch on the RTX 4050 Laptop GPU (Phase 1), ~25 minutes total for 30
epochs. Inference/evaluation on the 128-sample test set: well under 1 minute.

**KNOWN ISSUES:**
- Training loss and validation IoU were still improving at epoch 30 (last epoch's val_iou=0.6073
  was close to but slightly below the epoch-29 best of 0.6103) — the model had likely not fully
  converged within the 30-epoch budget. This is an intentionally scoped baseline run, not a
  claim of a fully tuned/converged model; Phase 8 (research experiments) is where tuning/longer
  training is considered if it provides measurable value.
- The baseline's simple channel-concatenation design (vs. a true Siamese shared encoder) is a
  known architectural limitation, not a bug — it is the documented reason Phase 5 exists.
- No hyperparameter search was performed; `configs/baseline.yaml` values are reasonable defaults,
  not the result of tuning.

**NEXT PHASE:**
- PHASE 5 — Siamese U-Net: implement `models/siamese_encoder.py` and `models/siamese_unet.py`
  with a shared encoder between before/after branches and configurable feature comparison
  (absolute difference / concatenation / both), verify forward/backward/optimizer-step, then train
  and evaluate it with the same `Trainer`/`evaluate.py` pipeline built in Phase 4 for a real,
  apples-to-apples comparison against this baseline.

---

## PHASE 3 — Data Preprocessing & DataLoader

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/data/preprocessing.py`: image/mask loading (OpenCV), mask binarization (explicit
  threshold at 127, addressing the non-binary mask values found during Phase 2 verification),
  resize (linear for images, nearest-neighbor for masks to preserve binary values), [0,1]
  normalization, and numpy->CHW-float-tensor conversion.
- `src/data/augmentation.py`: `PairedAugmentor` applying identical spatial transforms (horizontal
  flip, vertical flip, 90-degree rotation, random-crop+resize "scale" jitter) to the before image,
  after image, and mask from one shared random draw per call, plus independent brightness jitter
  applied to the before/after images only (never the mask).
- `src/data/dataset.py`: `LEVIRCDDataset(Dataset)` — verifies A/B/label pairing at construction
  time (raises `ValueError` listing unpaired files, or `FileNotFoundError` if the split directory
  is missing), returns `(before, after, mask)` tensors per the pipeline above.
- `src/data/dataloader.py`: `get_dataloader()` wrapping `LEVIRCDDataset` in a `torch.utils.data.
  DataLoader`, defaulting to shuffle+augment on for `train` and off for `val`/`test`.
- `scripts/verify_dataloader.py`: end-to-end Phase 3 verification against the real LEVIR-CD data
  (train/val/test `dataset[0]`, an augmented sample, DataLoader batch shapes, and an augmentation
  visual sanity check grid).
- `tests/` pytest suite (16 tests) using small synthetic images (per the rule to avoid requiring
  the full multi-GB dataset for unit tests): `test_preprocessing.py`, `test_augmentation.py`
  (including a same-pixel-tracking test proving spatial augmentation moves the mask in lockstep
  with the images), `test_dataset.py` (pairing-error and missing-split-dir error paths, correct
  tensor shapes/dtypes, DataLoader batch shapes and train/val augment-default behavior).

**FILES CREATED:**
- `src/__init__.py`, `src/data/__init__.py`
- `src/data/preprocessing.py`, `src/data/augmentation.py`, `src/data/dataset.py`,
  `src/data/dataloader.py`
- `scripts/verify_dataloader.py`
- `tests/conftest.py`, `tests/test_preprocessing.py`, `tests/test_augmentation.py`,
  `tests/test_dataset.py`
- `outputs/visualizations/augmentation_samples.png` (gitignored)

**FILES MODIFIED:**
- `requirements.txt` (added `pytest`, `huggingface_hub`)

**COMMANDS EXECUTED:**
- `pip install pytest`
- `venv/Scripts/python.exe scripts/verify_dataloader.py`
- `venv/Scripts/python.exe -m pytest tests/ -v`

**TESTS:**
- `scripts/verify_dataloader.py` against the real dataset: asserts tensor shapes
  `(3,256,256)`/`(3,256,256)`/`(1,256,256)`, dtype `float32`, image value range `[0,1]`, and mask
  values `⊆ {0,1}` for `train[0]`, `val[0]`, `test[0]`, and an augmented `train[0]`; asserts
  DataLoader batch shapes for `train` (batch_size=4, drop_last=True, 111 batches) and `val`
  (batch_size=4, 16 batches); renders and visually inspected an augmentation sample grid.
- `pytest tests/` — 16/16 passed, covering mask binarization thresholding, normalization range,
  resize dimension/interpolation-mode correctness, tensor shape/dtype conversion, paired-spatial-
  augmentation correctness (marker-pixel tracking), zero-probability no-op augmentation, shape
  preservation under scale jitter, dataset length/pairing/error-path correctness, and DataLoader
  batch shapes and augment defaults.

**RESULTS (actual, measured):**
```
Real-data verification (scripts/verify_dataloader.py): ALL CHECKS PASSED
  train[0]: before/after=(3,256,256) float32 in [0,1], mask=(1,256,256) float32 {0,1} - OK
  val[0], test[0]: same shape/dtype/range checks - OK
  train[0] with augmentation: same shape/dtype/range checks - OK
  DataLoader(train, batch_size=4): 111 batches, shapes (4,3,256,256)/(4,3,256,256)/(4,1,256,256) - OK
  DataLoader(val, batch_size=4): 16 batches - OK
pytest tests/: 16 passed, 0 failed, 2.17s
```
Augmentation visual check confirmed by direct image inspection: flips/rotations/scale-jitter move
the mask's changed region in exact lockstep with the before/after images across 5 sampled
variants — no misalignment observed.

**KNOWN ISSUES:**
- `num_workers=0` used by default in `get_dataloader` (single-process loading) — adequate at
  current dataset size/throughput; can be revisited in Phase 6 if training-time data loading
  becomes a bottleneck.
- Default `image_size=256` (not the native 1024) chosen for compute/VRAM tractability on the
  6 GB GPU identified in Phase 1; this is a deliberate, documented choice, not a limitation
  discovered by accident.

**NEXT PHASE:**
- PHASE 4 — Baseline U-Net: implement `models/unet.py`, `models/losses.py` (BCE, Dice, BCE+Dice),
  evaluation metrics (IoU/Dice/Precision/Recall/F1/Accuracy), train the baseline on a
  single-fused-input formulation, and report real test-set metrics (never fabricated).

---

## PHASE 2 — Dataset Acquisition & Understanding

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Identified the official LEVIR-CD source (https://justchenhao.github.io/LEVIR/, Chen & Shi 2020)
  and confirmed its official download links (Google Drive / Baidu Drive) are manual-only —
  interactive cloud-drive folders with no stable unauthenticated direct-download URL, unsuitable
  for reliable unattended scripted download (Google Drive in particular gates large files behind
  a browser confirmation click).
- Per the stop-condition/acquisition rules in `DEVELOPMENT_RULES.md` / `PROJECT_CONTEXT.md`,
  located and used a documented mirror instead: Hugging Face dataset repo
  `satellite-image-deep-learning/LEVIR-CD`, which exposes `train.zip` / `val.zip` / `test.zip`
  as plain, individually addressable files downloadable via `huggingface_hub` with no
  authentication. This is documented as a mirror, not the original source, in `docs/DATASET.md`.
- Downloaded all three official splits (`val.zip` first to validate structure/legitimacy before
  committing to the larger downloads, then `test.zip`, then `train.zip`), verified each archive's
  byte size against the Hub-reported size before extraction (exact match on all three), and
  extracted them to `data/raw/levir_cd/{train,val,test}/{A,B,label}/`.
- Wrote `scripts/verify_dataset.py`: checks A/B/label pairing (missing/orphan detection),
  opens and decodes every image (corruption check), verifies dimensions/channels/format
  consistency, computes the changed-vs-unchanged pixel distribution, and renders a 4-column
  (before/after/mask/overlay) sample visualization grid per split.
- Ran full verification on all three splits; wrote `docs/DATASET.md` with the real measured
  results (source, mirror justification, structure, per-split counts, class-imbalance numbers,
  a mask-binarization caveat, and split/leakage methodology).
- Deleted the now-redundant zip archives after verified extraction (2.3 GB reclaimed); dataset
  now lives only as extracted files under `data/raw/levir_cd/` (gitignored — not committed).

**FILES CREATED:**
- `scripts/verify_dataset.py`
- `docs/DATASET.md`
- `outputs/metrics/dataset_verification.json` (full machine-readable verification report, gitignored)
- `outputs/visualizations/dataset_samples_{train,val,test}.png` (gitignored)
- `data/raw/levir_cd/{train,val,test}/{A,B,label}/*.png` (dataset itself, gitignored)

**FILES MODIFIED:**
- None outside the above (README/DEVELOPMENT_LOG updates are part of this same entry)

**COMMANDS EXECUTED:**
- `pip install huggingface_hub`
- `HfApi().list_repo_files(...)` / `repo_info(..., files_metadata=True)` — file listing and size check
- `hf_hub_download(...)` for `val.zip`, `test.zip`, `train.zip` (run sequentially, each in the
  background due to slow connection speed — observed ~250 KB/s)
- `python -c "zipfile...extractall(...)"` for each split
- `venv/Scripts/python.exe scripts/verify_dataset.py --root data/raw/levir_cd --splits val test train`
- `rm -rf data/raw/levir_cd_zips`

**TESTS:**
- Downloaded file byte sizes compared against Hub-reported sizes — exact match for all three
  (`val.zip` 246,152,048; `test.zip` 496,305,323; `train.zip` 1,721,956,862 bytes).
- `scripts/verify_dataset.py` run against all three splits: pairing check, corruption check
  (every file actually opened/decoded), dimension/channel/format consistency check, pixel-value
  range check, pixel-distribution computation.
- Manually inspected the generated sample-grid visualizations for all three splits — change masks
  visibly align with genuine new construction between before/after images in every sampled case.

**RESULTS (actual, measured — full report in `outputs/metrics/dataset_verification.json`):**
```
train: 445 samples, 445/445/445 paired, 0 corrupted, all 1024x1024, changed-pixel fraction 0.04589
val:    64 samples,  64/64/64  paired, 0 corrupted, all 1024x1024, changed-pixel fraction 0.04197
test:  128 samples, 128/128/128 paired, 0 corrupted, all 1024x1024, changed-pixel fraction 0.05094
total: 637 samples — exact match to the official LEVIR-CD dataset size (637 pairs)
```
Mask pixel values are not perfectly binary in all files (some contain anti-aliased edge values
like 156/254 alongside 0/255) — flagged for explicit thresholding in Phase 3 preprocessing.

**KNOWN ISSUES:**
- Download speed from the Hugging Face mirror was slow (~250 KB/s observed, unauthenticated
  requests); downloads took roughly 15 min (val), 30 min (test), and just over an hour (train).
  Not a correctness issue, just a environment/network characteristic worth knowing for future
  re-downloads.
- The dataset's official split is used as-is (no re-splitting performed), per the leakage-
  prevention methodology documented in `docs/DATASET.md`.

**NEXT PHASE:**
- PHASE 3 — Data Preprocessing & DataLoader: implement `src/data/dataset.py`,
  `src/data/preprocessing.py` (including mask binarization per the caveat above),
  `src/data/augmentation.py` (paired spatial augmentation applied identically to A/B/label), and
  `src/data/dataloader.py`; verify `dataset[0]` returns correctly-shaped/typed tensors and that
  DataLoader batching works before any model code is written.

---

## PHASE 1 — Environment & Project Setup

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Created a Python virtual environment at `venv/` (Python 3.13.0).
- Installed PyTorch 2.6.0 + torchvision 0.21.0 with CUDA 12.4 wheels
  (`--index-url https://download.pytorch.org/whl/cu124`), matched to the detected NVIDIA GPU.
- Installed core stack: NumPy, Pandas, OpenCV, Pillow, scikit-learn, PyYAML, Matplotlib, Plotly,
  python-dotenv, tqdm.
- Installed Streamlit (dashboard dependency, installed now for environment completeness; the
  dashboard itself is not built until Phase 10 per the project rules).
- Wrote `scripts/check_env.py`, an environment diagnostic script that reports Python/PyTorch/
  torchvision versions, CUDA availability, GPU name/VRAM, and runs a real tensor matmul on both
  the selected device and an explicit CPU fallback.
- Pinned exact installed versions into `requirements.txt` (replacing the Phase 0 unpinned
  placeholder), with a documented CPU-vs-CUDA install note.

**FILES CREATED:**
- `venv/` (virtual environment, gitignored)
- `scripts/check_env.py`
- `outputs/env_freeze.txt` (full `pip freeze` output, local diagnostic artifact, gitignored)

**FILES MODIFIED:**
- `requirements.txt` (unpinned placeholder -> exact pinned versions from the verified environment)

**COMMANDS EXECUTED:**
- `python -m venv venv`
- `venv/Scripts/python.exe -m pip install --upgrade pip`
- `venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- `venv/Scripts/python.exe -m pip install numpy pandas opencv-python Pillow scikit-learn PyYAML matplotlib plotly python-dotenv tqdm`
- `venv/Scripts/python.exe -m pip install streamlit`
- `nvidia-smi` (GPU presence check, prior to choosing the CUDA wheel)
- `venv/Scripts/python.exe scripts/check_env.py`
- `venv/Scripts/python.exe -m pip freeze > outputs/env_freeze.txt`

**TESTS:**
- Ran `scripts/check_env.py` end-to-end; it imports torch/torchvision/numpy/cv2/PIL, checks
  `torch.cuda.is_available()`, enumerates GPU properties, runs `matmul` on the selected device,
  and runs a second `matmul` explicitly pinned to CPU to confirm CPU fallback works independent
  of GPU presence.
- Verified `streamlit` imports and reports its version.

**RESULTS (actual, measured — see `scripts/check_env.py` output):**
```
Python:      3.13.0
PyTorch:     2.6.0+cu124
Torchvision: 0.21.0+cu124
CUDA available: True
CUDA version (torch build): 12.4
GPU count: 1
GPU 0: NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB total VRAM)
Selected device: cuda
NumPy: 2.5.2
OpenCV: 5.0.0
Pillow: 12.3.0
Tensor op test on cuda: matmul(4x4, 4x4) -> shape (4, 4), sum=12.9776
Tensor op test on cpu (explicit fallback check): shape (4, 4), sum=12.2680
streamlit: 1.62.0
```

**KNOWN ISSUES:**
- GPU has 6 GB total VRAM, of which ~2.5 GB was already in use by an unrelated local process
  (an Ollama server) at the time of this check, leaving roughly 3.5 GB free. This constrains
  batch size / image resolution for later training phases and will be revisited in Phase 6
  when real batch sizes are chosen.
- `outputs/env_freeze.txt` and `venv/` are local artifacts (gitignored), not committed.

**NEXT PHASE:**
- PHASE 2 — Dataset Acquisition & Understanding: obtain the LEVIR-CD benchmark dataset, verify
  structure/pairing/dimensions/channels/label values, check for missing/corrupted files, compute
  changed-vs-unchanged pixel distribution, document splitting methodology (leakage prevention),
  and write `docs/DATASET.md`.

---

## PHASE 0 — Project Initialization

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Full directory scaffold created (`data/`, `models/`, `src/*`, `dashboard/*`, `notebooks/`,
  `configs/`, `outputs/*`, `tests/`, `docs/`, `scripts/`).
- Core documentation created: `PROJECT_CONTEXT.md`, `DEVELOPMENT_RULES.md`, `DEVELOPMENT_LOG.md`,
  `README.md`.
- `.gitignore` and `requirements.txt` (placeholder, to be pinned in Phase 1) created.
- `.env.example` created.
- Git repository initialized with an initial commit.

**FILES CREATED:**
- `PROJECT_CONTEXT.md`
- `DEVELOPMENT_RULES.md`
- `DEVELOPMENT_LOG.md`
- `README.md`
- `.gitignore`
- `requirements.txt`
- `.env.example`
- Directory scaffold (empty dirs hold `.gitkeep` where needed)

**FILES MODIFIED:**
- None (fresh repository)

**COMMANDS EXECUTED:**
- `mkdir -p` for the full directory tree
- `git init`
- `git add` / `git commit` (see git log for exact commit)

**TESTS:**
- Verified directory tree with `find . -maxdepth 3 -type d`.
- Verified `git status` / `git log` show a clean initial commit.
- Verified Python 3.13.0 is available on PATH (`python --version`).

**RESULTS:**
- Repository structure matches the target layout in `PROJECT_CONTEXT.md`.
- Git initialized and working.
- Python 3.13.0 detected. PyTorch/CUDA/GPU/etc. NOT YET MEASURED — deferred to Phase 1.

**KNOWN ISSUES:**
- None.

**NEXT PHASE:**
- PHASE 1 — Environment & Project Setup: create virtual environment, install pinned dependencies,
  verify PyTorch/torchvision/CUDA, produce environment diagnostic report.
