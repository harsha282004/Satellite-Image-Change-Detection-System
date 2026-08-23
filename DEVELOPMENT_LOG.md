# DEVELOPMENT_LOG.md

Running log of phase completions. Newest entry at the top. See `PROJECT_CONTEXT.md` for phase
definitions and `DEVELOPMENT_RULES.md` for the verification rules each entry must satisfy.

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
