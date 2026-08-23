"""Environment diagnostic for satellite-change-detection.

Run with: venv/Scripts/python.exe scripts/check_env.py
Verifies the core Deep Learning stack (PyTorch/torchvision/CUDA) is usable,
falls back to CPU when no GPU is available, and exercises a real tensor op.
"""
import platform
import sys


def main() -> int:
    print("=== Environment Diagnostic Report ===")
    print(f"Python:      {platform.python_version()} ({sys.executable})")

    try:
        import torch
    except ImportError as e:
        print(f"PyTorch:     NOT INSTALLED ({e})")
        return 1
    print(f"PyTorch:     {torch.__version__}")

    try:
        import torchvision
        print(f"Torchvision: {torchvision.__version__}")
    except ImportError as e:
        print(f"Torchvision: NOT INSTALLED ({e})")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version (torch build): {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB total VRAM)")
        device = torch.device("cuda")
    else:
        print("GPU: None detected / CUDA not available")
        device = torch.device("cpu")

    print(f"Selected device: {device}")

    for name, mod in (("NumPy", "numpy"), ("OpenCV", "cv2"), ("Pillow", "PIL")):
        try:
            m = __import__(mod)
            version = getattr(m, "__version__", None)
            if version is None and mod == "PIL":
                from PIL import __version__ as version  # type: ignore
            print(f"{name}: {version}")
        except ImportError as e:
            print(f"{name}: NOT INSTALLED ({e})")

    # Real tensor op on the selected device (GPU if available, else CPU).
    a = torch.rand(4, 4, device=device)
    b = torch.rand(4, 4, device=device)
    c = a @ b
    print(f"Tensor op test on {device}: matmul(4x4, 4x4) -> shape {tuple(c.shape)}, "
          f"sum={c.sum().item():.4f}")

    # Explicit CPU fallback test, independent of what the default device is.
    a_cpu = torch.rand(4, 4, device="cpu")
    b_cpu = torch.rand(4, 4, device="cpu")
    c_cpu = a_cpu @ b_cpu
    print(f"Tensor op test on cpu (explicit fallback check): shape {tuple(c_cpu.shape)}, "
          f"sum={c_cpu.sum().item():.4f}")

    print("=== Diagnostic complete: environment is usable ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
