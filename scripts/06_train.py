"""
06_train.py
===========
YOLO11m Person Detection Pipeline — Step 6: Training

Loads the pretrained yolo11m.pt checkpoint and launches fine-tuning
with all hyperparameters specified in the training notes (Section 5).

GPU: Single GPU, device=0 (hardcoded, confirmed).
CLI overrides available for: batch, workers, imgsz, epochs, patience.

Usage:
    python scripts/06_train.py
    python scripts/06_train.py --batch 8 --imgsz 640  # hardware flexibility
    python scripts/06_train.py --resume  # resume from last checkpoint
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_YAML = CONFIG_DIR / "data.yaml"
PRETRAINED_MODEL = PROJECT_ROOT / "yolo11m.pt"

# Fixed training hyperparameters from the training notes (Section 5)
TRAINING_DEFAULTS = {
    "data": None,                    # Set at runtime (absolute path to data.yaml)
    "epochs": 150,
    "imgsz": 960,                    # Higher res for small/distant person detection
    "batch": 16,
    "patience": 30,                  # Early stopping patience
    "device": 0,                     # Single GPU, confirmed
    "workers": 8,
    "optimizer": "auto",
    "cos_lr": True,                  # Cosine learning rate schedule
    "close_mosaic": 10,              # Disable mosaic for last 10 epochs
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,              # Boosts effective crowd density
    "degrees": 5.0,                  # Rotation augmentation
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "project": str(PROJECT_ROOT / "innovision_person_detection"),
    "name": "yolo11m_fbox_v1",
}


def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def verify_prerequisites(args: argparse.Namespace) -> bool:
    """Verify all prerequisites are in place before training."""
    ok = True

    data_yaml_path = Path(args.data) if args.data else DATA_YAML

    # Check data.yaml
    if not data_yaml_path.exists():
        print(f"  ✗ data.yaml not found at: {data_yaml_path}")
        print(f"    Run scripts/04_split_and_organize.py (or 04b) first.")
        ok = False
    else:
        print(f"  ✓ data.yaml: {data_yaml_path}")

    # Check pretrained model
    if not PRETRAINED_MODEL.exists():
        print(f"  ⚠ yolo11m.pt not found at: {PRETRAINED_MODEL}")
        print(f"    Ultralytics will auto-download it during training.")
    else:
        size_mb = PRETRAINED_MODEL.stat().st_size / (1024**2)
        print(f"  ✓ yolo11m.pt: {PRETRAINED_MODEL} ({size_mb:.1f} MB)")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print(f"  ⚠ No CUDA GPU detected — training will be VERY slow on CPU")
    except ImportError:
        print(f"  ✗ PyTorch not found")
        ok = False

    # Check dataset directories (only checking if default dataset/ is used, otherwise assume data.yaml is correct)
    if not args.data:
        for split in ["train", "val"]:
            img_dir = PROJECT_ROOT / "dataset" / "images" / split
            lbl_dir = PROJECT_ROOT / "dataset" / "labels" / split
            if img_dir.exists() and lbl_dir.exists():
                img_count = len(list(img_dir.iterdir()))
                lbl_count = len(list(lbl_dir.glob("*.txt")))
                print(f"  ✓ {split}: {img_count:,} images, {lbl_count:,} labels")
            else:
                print(f"  ✗ {split} split not found")
                ok = False

    return ok


def run_training(args: argparse.Namespace) -> None:
    """Launch YOLO11m training."""
    from ultralytics import YOLO

    # Determine model path
    if args.resume:
        # Resume from last checkpoint
        run_name = args.name if args.name else "yolo11m_fbox_v1"
        last_pt = PROJECT_ROOT / "innovision_person_detection" / run_name / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"  ✗ Cannot resume — last.pt not found at: {last_pt}")
            sys.exit(1)
        print(f"  → Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model_path = str(PRETRAINED_MODEL) if PRETRAINED_MODEL.exists() else "yolo11m.pt"
        print(f"  → Loading model: {model_path}")
        model = YOLO(model_path)

    # Build training config with overrides
    train_config = TRAINING_DEFAULTS.copy()
    train_config["data"] = str(Path(args.data).resolve()) if args.data else str(DATA_YAML)

    # Apply CLI overrides
    if args.batch is not None:
        train_config["batch"] = args.batch
    if args.imgsz is not None:
        train_config["imgsz"] = args.imgsz
    if args.workers is not None:
        train_config["workers"] = args.workers
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.patience is not None:
        train_config["patience"] = args.patience
    if args.name is not None:
        train_config["name"] = args.name

    # Print training configuration
    print_header("Training Configuration")
    for key, val in sorted(train_config.items()):
        print(f"  {key:20s}: {val}")

    # Launch training
    print_header("Starting Training")
    print(f"  This will take ~20-30 hours on an A100 at imgsz={train_config['imgsz']}")
    print(f"  Monitor progress in: {train_config['project']}/{train_config['name']}/")
    print()

    if args.resume:
        results = model.train(resume=True)
    else:
        results = model.train(**train_config)

    # Training complete
    print_header("Training Complete")
    run_dir = Path(train_config["project"]) / train_config["name"]
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print(f"  Run directory : {run_dir}")
    print(f"  best.pt       : {best_pt} ({'✓ exists' if best_pt.exists() else '✗ not found'})")
    print(f"  last.pt       : {last_pt} ({'✓ exists' if last_pt.exists() else '✗ not found'})")
    print()
    print("  Next step: python scripts/07_validate.py")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO11m Person Detection — Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/06_train.py                      # Default config
  python scripts/06_train.py --batch 8             # Smaller batch for limited VRAM
  python scripts/06_train.py --imgsz 640           # Lower resolution for faster training
  python scripts/06_train.py --resume              # Resume from last checkpoint
        """,
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (default: 960)")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers (default: 8)")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs (default: 150)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: 30)")
    parser.add_argument("--data", type=str, default=None, help="Path to custom data.yaml")
    parser.add_argument("--name", type=str, default=None, help="Custom run name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    print_header("YOLO11m Person Detection — Training Launch")

    # Pre-flight checks
    print_header("Pre-flight Checks")
    if not verify_prerequisites(args):
        print("\n  ✗ Prerequisites not met. Fix the issues above and try again.")
        sys.exit(1)

    print("\n  ✓ All prerequisites verified.")
    run_training(args)
