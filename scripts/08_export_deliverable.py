"""
08_export_deliverable.py
========================
YOLO11m Person Detection Pipeline — Step 8: Export Deliverable

Packages the trained model, metrics, and configuration into a clean
deliverable/ folder structure with an auto-generated README.

Output structure:
    deliverable/
    ├── model/
    │   ├── best.pt          # Best validation checkpoint (THE deliverable)
    │   └── last.pt          # Final epoch checkpoint (for resume)
    ├── metrics/
    │   ├── results.csv      # Per-epoch training metrics
    │   ├── results.png      # Training curves plot
    │   ├── confusion_matrix.png
    │   ├── confusion_matrix_normalized.png
    │   ├── F1_curve.png
    │   ├── P_curve.png
    │   ├── R_curve.png
    │   ├── PR_curve.png
    │   └── validation_metrics.json  # Per-source validation results
    ├── config/
    │   ├── data.yaml        # Dataset configuration used
    │   └── args.yaml        # Training arguments used
    └── README.md            # Auto-generated training summary

Usage:
    python scripts/08_export_deliverable.py
    python scripts/08_export_deliverable.py --run-dir path/to/training/run
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLE_DIR = PROJECT_ROOT / "deliverable"

# Default training run directory
DEFAULT_RUN_DIR = PROJECT_ROOT / "innovision_person_detection" / "yolo11m_fbox_v1"

# Files to copy from the training run
WEIGHT_FILES = ["best.pt", "last.pt"]
METRIC_FILES = [
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "PR_curve.png",
]
CONFIG_FILES = ["args.yaml"]


def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def safe_copy(src: Path, dst: Path) -> bool:
    """Copy a file if source exists. Returns True if copied."""
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        size_mb = src.stat().st_size / (1024**2)
        print(f"  ✓ {dst.relative_to(DELIVERABLE_DIR)} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ⚠ Not found: {src.name}")
        return False


def copy_weights(run_dir: Path) -> dict:
    """Copy model weights to deliverable."""
    print_header("Copying Model Weights")
    weights_dir = run_dir / "weights"
    results = {}

    for filename in WEIGHT_FILES:
        src = weights_dir / filename
        dst = DELIVERABLE_DIR / "model" / filename
        results[filename] = safe_copy(src, dst)

    return results


def copy_metrics(run_dir: Path) -> dict:
    """Copy metrics/plots to deliverable."""
    print_header("Copying Metrics & Plots")
    results = {}

    for filename in METRIC_FILES:
        src = run_dir / filename
        dst = DELIVERABLE_DIR / "metrics" / filename
        results[filename] = safe_copy(src, dst)

    # Also copy validation metrics JSON if it exists
    val_metrics = PROJECT_ROOT / "config" / "validation_metrics.json"
    if val_metrics.exists():
        dst = DELIVERABLE_DIR / "metrics" / "validation_metrics.json"
        safe_copy(val_metrics, dst)
        results["validation_metrics.json"] = True

    return results


def copy_config(run_dir: Path) -> dict:
    """Copy configuration files to deliverable."""
    print_header("Copying Configuration")
    results = {}

    # Copy training args
    for filename in CONFIG_FILES:
        src = run_dir / filename
        dst = DELIVERABLE_DIR / "config" / filename
        results[filename] = safe_copy(src, dst)

    # Copy data.yaml
    data_yaml = PROJECT_ROOT / "config" / "data.yaml"
    dst = DELIVERABLE_DIR / "config" / "data.yaml"
    results["data.yaml"] = safe_copy(data_yaml, dst)

    return results


def load_validation_metrics() -> dict:
    """Load validation metrics if available."""
    metrics_path = PROJECT_ROOT / "config" / "validation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


def load_training_args(run_dir: Path) -> dict:
    """Load training args.yaml if available."""
    args_path = run_dir / "args.yaml"
    if args_path.exists():
        import yaml
        with open(args_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def format_metric(val, fmt: str = ".4f") -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def generate_readme(run_dir: Path) -> None:
    """Auto-generate README.md for the deliverable."""
    print_header("Generating README.md")

    val_metrics = load_validation_metrics()
    train_args = load_training_args(run_dir)
    overall = val_metrics.get("overall", {})
    per_source = val_metrics.get("per_source", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build README content
    readme = f"""# YOLO11m Person Detection — Trained Model Deliverable

**Project:** Innovision Multi-Analytics Security Platform — Person Detection Module
**Generated:** {timestamp}
**Phase:** Phase 1 (COCO + CrowdHuman, fbox only)

---

## Quick Start

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("model/best.pt")

# Run inference on an image
results = model.predict("path/to/image.jpg", conf=0.25)

# Run inference on a video
results = model.predict("path/to/video.mp4", conf=0.25, stream=True)

# Run inference on a camera stream
results = model.predict(0, conf=0.25, stream=True)  # webcam
```

---

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLO11m (fine-tuned) |
| Base model | yolo11m.pt (pretrained on COCO) |
| Classes | 1 (person) |
| Input size | {train_args.get('imgsz', 960)}×{train_args.get('imgsz', 960)} |
| Export format | .pt (PyTorch/Ultralytics) |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | {train_args.get('epochs', 150)} |
| Image size | {train_args.get('imgsz', 960)} |
| Batch size | {train_args.get('batch', 16)} |
| Optimizer | {train_args.get('optimizer', 'auto')} |
| Cosine LR | {train_args.get('cos_lr', True)} |
| Patience | {train_args.get('patience', 30)} |
| Close mosaic | {train_args.get('close_mosaic', 10)} |
| Copy-paste | {train_args.get('copy_paste', 0.1)} |
| Mixup | {train_args.get('mixup', 0.1)} |
| Device | GPU 0 (single GPU) |

---

## Datasets Used (Phase 1)

| Dataset | Role | Images | Person instances |
|---|---|---|---|
| COCO (person class) | General person detection | ~64,000 | ~262,000 |
| CrowdHuman (fbox) | Crowded/occluded scenes | ~19,000 | ~470,000 |
| **Total** | | **~83,000** | **~732,000** |

Split ratio: 70% train / 20% val / 10% test (applied per source dataset).

---

## Validation Results

### Overall (test split)

| Metric | Value |
|---|---|
| mAP50 | {format_metric(overall.get('mAP50'))} |
| mAP50-95 | {format_metric(overall.get('mAP50-95'))} |
| Precision | {format_metric(overall.get('precision'))} |
| Recall | {format_metric(overall.get('recall'))} |

### Per-Source Breakdown

| Source | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Overall | {format_metric(overall.get('mAP50'))} | {format_metric(overall.get('mAP50-95'))} | {format_metric(overall.get('precision'))} | {format_metric(overall.get('recall'))} |"""

    for source_name, source_metrics in per_source.items():
        if "error" in source_metrics:
            readme += f"\n| {source_name.upper()} | N/A | N/A | N/A | N/A |"
        else:
            readme += (
                f"\n| {source_name.upper()} "
                f"| {format_metric(source_metrics.get('mAP50'))} "
                f"| {format_metric(source_metrics.get('mAP50-95'))} "
                f"| {format_metric(source_metrics.get('precision'))} "
                f"| {format_metric(source_metrics.get('recall'))} |"
            )

    readme += f"""

---

## Folder Structure

```
deliverable/
├── model/
│   ├── best.pt              # ← THE DELIVERABLE — best validation checkpoint
│   └── last.pt              # Final epoch checkpoint (for resuming training)
├── metrics/
│   ├── results.csv          # Per-epoch training metrics
│   ├── results.png          # Training curves visualization
│   ├── confusion_matrix.png # Confusion matrix
│   ├── F1_curve.png         # F1 vs confidence threshold
│   ├── P_curve.png          # Precision vs confidence
│   ├── R_curve.png          # Recall vs confidence
│   ├── PR_curve.png         # Precision-Recall curve
│   └── validation_metrics.json  # Detailed per-source validation results
├── config/
│   ├── data.yaml            # Dataset configuration
│   └── args.yaml            # Training arguments
└── README.md                # This file
```

---

## Phase 2 Extension Notes

This model is **Phase 1** — trained on COCO + CrowdHuman only. Planned extensions:

1. **WiderPerson (Phase 2)**: Adds surveillance-angle/wide-view images (~8k images). The current model may be weaker on surveillance camera perspectives until this data is added.
2. **vbox comparison**: Currently trained on fbox (full-body boxes). vbox (visible-only boxes) training can be done later if occlusion-scene performance needs investigation.
3. **Export formats**: The .pt checkpoint can be exported to .onnx (cross-platform) or .engine/TensorRT (NVIDIA-optimized) at any time if deployment requirements change.

---

## How to Resume Training

```python
from ultralytics import YOLO

model = YOLO("model/last.pt")
results = model.train(resume=True)
```

---

## How to Export to Other Formats

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")

# Export to ONNX
model.export(format="onnx")

# Export to TensorRT (requires NVIDIA GPU + TensorRT SDK)
model.export(format="engine")
```
"""

    readme_path = DELIVERABLE_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)

    print(f"  ✓ README.md generated: {readme_path}")


def print_final_summary() -> None:
    """Print the final deliverable summary."""
    print_header("Deliverable Package Complete")
    print(f"  Output directory: {DELIVERABLE_DIR}")
    print()

    # List all files in deliverable
    total_size = 0
    for root, dirs, files in os.walk(DELIVERABLE_DIR):
        for f in files:
            fp = Path(root) / f
            size_mb = fp.stat().st_size / (1024**2)
            total_size += size_mb
            rel = fp.relative_to(DELIVERABLE_DIR)
            print(f"  {str(rel):<50s} {size_mb:>8.1f} MB")

    print(f"\n  {'Total':50s} {total_size:>8.1f} MB")
    print()
    print("  ═══════════════════════════════════════════════")
    print("  ✓ Pipeline complete!")
    print("  ✓ Hand off deliverable/model/best.pt as the trained model.")
    print("  ═══════════════════════════════════════════════")
    print()


# Needed for os.walk
import os

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11m Person Detection — Export Deliverable")
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help=f"Training run directory (default: {DEFAULT_RUN_DIR})"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR

    print_header("YOLO11m Person Detection — Export Deliverable")

    # Verify run directory exists
    if not run_dir.exists():
        print(f"  ✗ Training run directory not found: {run_dir}")
        print(f"    Run scripts/06_train.py first, or specify --run-dir path/to/run")
        exit(1)

    print(f"  Run directory : {run_dir}")
    print(f"  Output        : {DELIVERABLE_DIR}")

    # Create clean deliverable directory
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy everything
    copy_weights(run_dir)
    copy_metrics(run_dir)
    copy_config(run_dir)
    generate_readme(run_dir)
    print_final_summary()
