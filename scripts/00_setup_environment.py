"""
00_setup_environment.py
=======================
YOLO11m Person Detection Pipeline — Step 0: Environment Setup

Installs required Python packages, verifies GPU/CUDA access,
creates the project folder structure, and downloads the yolo11m.pt
pretrained checkpoint.

Usage:
    python scripts/00_setup_environment.py

Re-runnable: safe to run multiple times (skips existing dirs/files).
"""

import subprocess
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_PACKAGES = [
    "ultralytics",
    "pycocotools",
    "opencv-python",
    "matplotlib",
    "gdown",
    "tqdm",
    "scikit-learn",
    "Pillow",
]

FOLDER_STRUCTURE = [
    "raw/coco/annotations",
    "raw/coco/images",
    "raw/crowdhuman/annotations",
    "raw/crowdhuman/images",
    "dataset/images/train",
    "dataset/images/val",
    "dataset/images/test",
    "dataset/labels/train",
    "dataset/labels/val",
    "dataset/labels/test",
    "config",
    "sanity_checks",
    "deliverable/model",
    "deliverable/metrics",
    "deliverable/config",
]


def print_header(msg: str) -> None:
    """Print a formatted section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def install_packages() -> None:
    """Install required Python packages via pip."""
    print_header("Installing required Python packages")
    for pkg in REQUIRED_PACKAGES:
        print(f"  → Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("  ✓ All packages installed successfully.")


def verify_gpu() -> None:
    """Verify CUDA/GPU access via PyTorch."""
    print_header("Verifying GPU / CUDA access")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA available  : {cuda_available}")
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  GPU device      : {device_name}")
            print(f"  VRAM            : {vram_gb:.1f} GB")
            print("  ✓ GPU is ready.")
        else:
            print("  ⚠ No CUDA GPU detected. Training will fall back to CPU (very slow).")
            print("    Make sure NVIDIA drivers and CUDA toolkit are installed.")
    except ImportError:
        print("  ✗ PyTorch not found. It should have been installed with ultralytics.")
        print("    Try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")


def create_folder_structure() -> None:
    """Create the project directory tree."""
    print_header("Creating folder structure")
    for folder in FOLDER_STRUCTURE:
        full_path = PROJECT_ROOT / folder
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {folder}/")
    print(f"\n  Project root: {PROJECT_ROOT}")


def download_pretrained_model() -> None:
    """Download yolo11m.pt pretrained checkpoint if not present."""
    print_header("Downloading yolo11m.pt pretrained checkpoint")
    model_path = PROJECT_ROOT / "yolo11m.pt"
    if model_path.exists():
        print(f"  ✓ Already exists: {model_path}")
        return

    try:
        from ultralytics import YOLO

        print("  → Downloading yolo11m.pt via Ultralytics (this may take a moment)...")
        # Loading the model name triggers auto-download of the checkpoint
        model = YOLO("yolo11m.pt")
        # Ultralytics downloads to the current working directory or its cache.
        # We move it to PROJECT_ROOT if it landed elsewhere.
        default_path = Path("yolo11m.pt")
        if default_path.exists() and default_path.resolve() != model_path.resolve():
            import shutil
            shutil.move(str(default_path), str(model_path))
        print(f"  ✓ Downloaded to {model_path}")
    except Exception as e:
        print(f"  ✗ Failed to download yolo11m.pt: {e}")
        print("    You can download it manually from the Ultralytics releases page.")


def print_summary() -> None:
    """Print a summary of what was set up."""
    print_header("Setup Complete — Summary")
    print(f"  Project root      : {PROJECT_ROOT}")
    print(f"  Scripts directory  : {PROJECT_ROOT / 'scripts'}")
    print(f"  Raw datasets       : {PROJECT_ROOT / 'raw'}")
    print(f"  Final dataset      : {PROJECT_ROOT / 'dataset'}")
    print(f"  Sanity checks      : {PROJECT_ROOT / 'sanity_checks'}")
    print(f"  Deliverable output : {PROJECT_ROOT / 'deliverable'}")
    print()
    print("  Next step: python scripts/01_download_datasets.py")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_header("YOLO11m Person Detection — Environment Setup")
    print(f"  Project root: {PROJECT_ROOT}")

    install_packages()
    verify_gpu()
    create_folder_structure()
    download_pretrained_model()
    print_summary()
