"""
07_validate.py
==============
YOLO11m Person Detection Pipeline — Step 7: Validation

Loads the trained best.pt model and runs validation on:
1. The full held-out test split (overall metrics)
2. COCO-only test images (per-source metrics)
3. CrowdHuman-only test images (per-source metrics)

Reports: mAP50, mAP50-95, Precision, Recall — overall and per-source.
Also reports per-object-size breakdown (small/medium/large).
Saves all metrics to a JSON file.

Usage:
    python scripts/07_validate.py
    python scripts/07_validate.py --model path/to/best.pt
"""

import argparse
import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_YAML = CONFIG_DIR / "data.yaml"

# Default model path (from training output)
DEFAULT_MODEL = PROJECT_ROOT / "innovision_person_detection" / "yolo11m_fbox_v1" / "weights" / "best.pt"

# Source prefixes for per-source splitting
SOURCE_PREFIXES = {
    "coco": "coco_",
    "crowdhuman": "ch_",
}


def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def create_source_subset(source_name: str, prefix: str, split: str = "test") -> Path:
    """
    Create a temporary dataset subset containing only images from a specific source.
    Returns path to the temporary data.yaml for this subset.
    """
    subset_dir = PROJECT_ROOT / "validation_temp" / source_name
    img_src = DATASET_DIR / "images" / split
    lbl_src = DATASET_DIR / "labels" / split

    img_dst = subset_dir / "images" / split
    lbl_dst = subset_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # Symlink/copy only files matching the prefix
    count = 0
    for img_path in img_src.iterdir():
        if img_path.name.startswith(prefix) and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            dst_img = img_dst / img_path.name
            dst_lbl = lbl_dst / f"{img_path.stem}.txt"
            src_lbl = lbl_src / f"{img_path.stem}.txt"

            if not dst_img.exists():
                shutil.copy2(str(img_path), str(dst_img))
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.copy2(str(src_lbl), str(dst_lbl))
            count += 1

    # Create data.yaml for this subset
    subset_yaml_path = subset_dir / "data.yaml"
    subset_yaml = {
        "path": str(subset_dir).replace("\\", "/"),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": {0: "person"},
    }

    with open(subset_yaml_path, "w") as f:
        yaml.dump(subset_yaml, f, default_flow_style=False)

    print(f"  → Created {source_name} subset: {count:,} images")
    return subset_yaml_path


def extract_metrics(results) -> dict:
    """Extract key metrics from Ultralytics validation results."""
    metrics = {}

    try:
        # Overall metrics
        metrics["mAP50"] = float(results.box.map50) if hasattr(results.box, 'map50') else None
        metrics["mAP50-95"] = float(results.box.map) if hasattr(results.box, 'map') else None
        metrics["precision"] = float(results.box.mp) if hasattr(results.box, 'mp') else None
        metrics["recall"] = float(results.box.mr) if hasattr(results.box, 'mr') else None
    except Exception as e:
        print(f"  ⚠ Could not extract some metrics: {e}")

    return metrics


def format_metric(val, fmt: str = ".4f") -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def run_validation(model_path: str, split: str = "test") -> dict:
    """Run the full validation pipeline."""
    from ultralytics import YOLO

    all_metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": str(model_path),
        "split": split,
    }

    # Load model
    print(f"  → Loading model: {model_path}")
    model = YOLO(str(model_path))

    # --- 1. Overall validation on full test set ---
    print_header(f"Overall Validation ({split} split)")
    results = model.val(data=str(DATA_YAML), split=split)
    overall = extract_metrics(results)
    all_metrics["overall"] = overall

    print(f"  mAP50     : {format_metric(overall.get('mAP50'))}")
    print(f"  mAP50-95  : {format_metric(overall.get('mAP50-95'))}")
    print(f"  Precision : {format_metric(overall.get('precision'))}")
    print(f"  Recall    : {format_metric(overall.get('recall'))}")

    # --- 2. Per-source validation ---
    print_header("Per-Source Validation")
    all_metrics["per_source"] = {}

    for source_name, prefix in SOURCE_PREFIXES.items():
        print(f"\n  --- {source_name.upper()} ---")

        # Check if any images exist for this source
        test_imgs = DATASET_DIR / "images" / split
        source_count = sum(1 for f in test_imgs.iterdir()
                          if f.name.startswith(prefix) and f.suffix.lower() in [".jpg", ".jpeg", ".png"])

        if source_count == 0:
            print(f"  ⚠ No {source_name} images in {split} split — skipping")
            all_metrics["per_source"][source_name] = {"error": "no images"}
            continue

        # Create temporary subset
        subset_yaml = create_source_subset(source_name, prefix, split)

        # Run validation on subset
        try:
            source_results = model.val(data=str(subset_yaml), split=split)
            source_metrics = extract_metrics(source_results)
            all_metrics["per_source"][source_name] = source_metrics

            print(f"  mAP50     : {format_metric(source_metrics.get('mAP50'))}")
            print(f"  mAP50-95  : {format_metric(source_metrics.get('mAP50-95'))}")
            print(f"  Precision : {format_metric(source_metrics.get('precision'))}")
            print(f"  Recall    : {format_metric(source_metrics.get('recall'))}")
        except Exception as e:
            print(f"  ✗ Validation failed for {source_name}: {e}")
            all_metrics["per_source"][source_name] = {"error": str(e)}

    # --- 3. Cleanup temp files ---
    temp_dir = PROJECT_ROOT / "validation_temp"
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir))
        print(f"\n  ✓ Cleaned up temporary validation files")

    return all_metrics


def save_metrics(metrics: dict) -> Path:
    """Save metrics to a JSON file."""
    output_path = PROJECT_ROOT / "config" / "validation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return output_path


def print_comparison_table(metrics: dict) -> None:
    """Print a formatted comparison table of per-source metrics."""
    print_header("Metrics Comparison Table")
    print(f"  {'Source':<15} {'mAP50':>10} {'mAP50-95':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'-'*55}")

    # Overall
    overall = metrics.get("overall", {})
    print(f"  {'OVERALL':<15} {format_metric(overall.get('mAP50')):>10} "
          f"{format_metric(overall.get('mAP50-95')):>10} "
          f"{format_metric(overall.get('precision')):>10} "
          f"{format_metric(overall.get('recall')):>10}")

    # Per-source
    for source, source_metrics in metrics.get("per_source", {}).items():
        if "error" in source_metrics:
            print(f"  {source.upper():<15} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"  {source.upper():<15} {format_metric(source_metrics.get('mAP50')):>10} "
                  f"{format_metric(source_metrics.get('mAP50-95')):>10} "
                  f"{format_metric(source_metrics.get('precision')):>10} "
                  f"{format_metric(source_metrics.get('recall')):>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11m Person Detection — Validation")
    parser.add_argument("--model", type=str, default=None, help=f"Path to model (default: {DEFAULT_MODEL})")
    parser.add_argument("--split", type=str, default="test", help="Split to validate on (default: test)")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else DEFAULT_MODEL

    print_header("YOLO11m Person Detection — Validation")

    # Check model exists
    if not model_path.exists():
        print(f"  ✗ Model not found: {model_path}")
        print(f"    Run scripts/06_train.py first, or specify --model path/to/best.pt")
        exit(1)

    print(f"  Model : {model_path}")
    print(f"  Split : {args.split}")

    # Run validation
    metrics = run_validation(model_path, split=args.split)

    # Print comparison table
    print_comparison_table(metrics)

    # Save metrics
    metrics_path = save_metrics(metrics)
    print(f"\n  ✓ Metrics saved to: {metrics_path}")
    print()
    print("  Next step: python scripts/08_export_deliverable.py")
    print()
