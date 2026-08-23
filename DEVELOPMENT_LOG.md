# DEVELOPMENT_LOG.md

Running log of phase completions. Newest entry at the top. See `PROJECT_CONTEXT.md` for phase
definitions and `DEVELOPMENT_RULES.md` for the verification rules each entry must satisfy.

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
